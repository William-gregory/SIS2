!> This code performs a state-dependent bias correction to each sea ice concentration category (part_size).
!> The bias correction method was developed by training a Machine Learning (ML) model to predict sea ice
!> concentration data assimilation increments, using model state variables. An initial CNN architecture is
!> used to find a mapping from model state variables to the aggregate (observable) SIC increment. This prediction
!> is then passed to an ANN, along with other state variables, to predict a correction to each category concentration.
!> Details of this approach can be found at: https://doi.org/10.1029/2023MS003757.

!> The implementation of this bias-correction framework was originally shown in https://doi.org/10.1029/2023GL106776.
!> However this approach was based on updating model restart files offline (high I/O). This present code applies the corrections
!> at the thermodynamic timestep of SIS2. Alternative versions of this code have been developed
!> for both FTorch and pure Fortran, and can be found at https://github.com/William-gregory/SIS2/tree/ftorch_SPEAR and
!> https://github.com/William-gregory/SIS2/tree/ML_pure_fortran, respectively.

!< Author: Will Gregory (wg4031@princeton.edu / william.gregory@noaa.gov)
!<
!<
!<

module SIS_ML

use ice_grid,                  only : ice_grid_type
use SIS_hor_grid,              only : SIS_hor_grid_type
use MOM_domains,               only : clone_MOM_domain,MOM_domain_type
use MOM_domains,               only : pass_var, pass_vector, CGRID_NE
use SIS_diag_mediator,         only : register_SIS_diag_field
use SIS_diag_mediator,         only : post_SIS_data, post_data=>post_SIS_data
use SIS_diag_mediator,         only : SIS_diag_ctrl
use SIS2_ice_thm,              only : get_SIS2_thermo_coefs
use SIS_types,                 only : ice_state_type, ocean_sfc_state_type, fast_ice_avg_type, ice_ocean_flux_type
use MOM_diag_mediator,         only : time_type
use MOM_file_parser,           only : get_param, param_file_type
use Forpy_interface,           only : forpy_run_python, python_interface

implicit none; private

!> Configure the SIS2 memory for halos required to pad the CNN
#include <SIS2_memory.h>
#ifdef STATIC_MEMORY_
#  ifndef BTHALO_
#    define BTHALO_ 0
#  endif
#  define WHALOI_ MAX(BTHALO_-NIHALO_,0)
#  define WHALOJ_ MAX(BTHALO_-NJHALO_,0)
#  define NIMEMW_   1-WHALOI_:NIMEM_+WHALOI_
#  define NJMEMW_   1-WHALOJ_:NJMEM_+WHALOJ_
#  define NIMEMBW_  -WHALOI_:NIMEM_+WHALOI_
#  define NJMEMBW_  -WHALOJ_:NJMEM_+WHALOJ_
#  define SZIW_(G)  NIMEMW_
#  define SZJW_(G)  NJMEMW_
#  define SZIBW_(G) NIMEMBW_
#  define SZJBW_(G) NJMEMBW_
#else
#  define NIMEMW_   :
#  define NJMEMW_   :
#  define NIMEMBW_  :
#  define NJMEMBW_  :
#  define SZIW_(G)  G%isdw:G%iedw
#  define SZJW_(G)  G%jsdw:G%jedw
#  define SZIBW_(G) G%isdw-1:G%iedw
#  define SZJBW_(G) G%jsdw-1:G%jedw
#endif

public :: ML_init,ML_inference

!> Control structure for ML model
type, public :: ML_CS ; private
  type(MOM_domain_type), pointer :: CNN_Domain => NULL()  !< Domain for inputs/outputs for CNN
  integer :: isdw !< The lower i-memory limit for the wide halo arrays.
  integer :: iedw !< The upper i-memory limit for the wide halo arrays.
  integer :: jsdw !< The lower j-memory limit for the wide halo arrays.
  integer :: jedw !< The upper j-memory limit for the wide halo arrays.
  integer :: CNN_halo_size  !< Halo size at each side of subdomains
  real    :: piston_SSTadj !< piston velocity of SST restoring
  character(len=300)  :: CNN_weights !< Optimized weights for the CNN
  character(len=300)  :: ANN_weights !< Optimized weights for the ANN

  type(SIS_diag_ctrl), pointer :: diag => NULL() !< A type that regulates diagnostics output
  !>@{ Diagnostic handles
  integer :: id_dCN = -1
  !>@}
  
end type ML_CS

contains

!> Initialize ML routine
subroutine ML_init(Time,G,param_file,diag,CS)
  type(time_type),               intent(in)    :: Time       !< The current model time.
  type(SIS_hor_grid_type),       intent(in)    :: G          !< The horizontal grid structure.
  type(param_file_type),         intent(in)    :: param_file !< Parameter file parser structure.
  type(SIS_diag_ctrl), target,   intent(inout) :: diag       !< Diagnostics structure.
  type(ML_CS),                  intent(inout)  :: CS         !< Control structure for the ML model(s)
  
  ! Local Variables
  integer :: wd_halos(2) ! Varies with CNN
  real, parameter :: missing = -1e34
  character(len=40)  :: mdl = "SIS_CNN"  ! module name

  ! Register fields for output from this module.
  CS%diag => diag

  call get_param(param_file, mdl, "CNN_HALO_SIZE", CS%CNN_halo_size, &
      "Halo size at each side of subdomains, depends on CNN architecture.", & 
      units="nondim", default=4)

  call get_param(param_file, mdl, "PISTON_SSTADJ", CS%piston_SSTadj, &
      "Piston velocity with which to restore SST after CNN correction", &
      units="m day-1", default=4.0)

  call get_param(param_file, mdl, "CNN_WEIGHTS", CS%CNN_weights, &
      "Optimized weights for the CNN which predicts dSIC", &
      default="/gpfs/f5/scratch/gfdl_o/William.Gregory/Forpy/NetworkA_weights_SPEAR.pt")

  call get_param(param_file, mdl, "ANN_WEIGHTS", CS%ANN_weights, &
      "Optimized weights for the ANN which predicts dCN", &
      default="/gpfs/f5/scratch/gfdl_o/William.Gregory/Forpy/NetworkA_weights_SPEAR.pt")
  
  wd_halos(1) = CS%CNN_halo_size
  wd_halos(2) = CS%CNN_halo_size
  if (G%symmetric) then
     call clone_MOM_domain(G%Domain, CS%CNN_Domain, min_halo=wd_halos, symmetric=.true.)
  else
     call clone_MOM_domain(G%Domain, CS%CNN_Domain, min_halo=wd_halos, symmetric=.false.)
  endif
  CS%isdw = G%isc-wd_halos(1) ; CS%iedw = G%iec+wd_halos(1)
  CS%jsdw = G%jsc-wd_halos(2) ; CS%jedw = G%jec+wd_halos(2)

end subroutine ML_init

!> This routine does all of the data prep for both the CNN and ANN, including padding the data
!> for the CNN, and normalizing all inputs. The predicted increments are the added to the prior
!> sea ice concentration states and a post-processing step then bounds this new (posterior) state
!> between 0 and 1, and then makes commensurate adjustments to the sea ice profiles in the case of
!> adding/removing sea ice (i.e add thickness and salinity for new ice). The code is currently non-
!> conservative in terms of heat, mass, salt.
subroutine ML_inference(IST, OSS, FIA, IOF, G, IG, CS, ML, dt_slow)
  type(ice_state_type),       intent(inout)  :: IST     !< A type describing the state of the sea ice
  type(fast_ice_avg_type),    intent(inout)  :: FIA     !< A type containing averages of fields
                                                        ! (mostly fluxes) over the fast updates
  type(ocean_sfc_state_type), intent(inout)  :: OSS     !< A structure containing the arrays that describe
                                                        !  the ocean's surface state for the ice model.
  type(ice_ocean_flux_type),  intent(inout)  :: IOF     !< A structure containing fluxes from the ice to
                                                        !  the ocean that are calculated by the ice model.
  type(SIS_hor_grid_type),    intent(in)     :: G       !< The horizontal grid structure
  type(ice_grid_type),        intent(in)     :: IG      !< Sea ice specific grid
  type(python_interface),     intent(in)     :: CS      !< Python interface object
  type(ML_CS) ,               intent(inout)  :: ML      !< Control structure for the ML model
  real,                       intent(in)     :: dt_slow !< The thermodynamic time step [T ~> s]

  real, dimension(SZIW_(ML),SZJW_(ML)) &
                                   ::  WH_SIC    !< aggregate concentrations [nondim].
  real, dimension(SZIW_(ML),SZJW_(ML)) &
                                   ::  WH_SST    !< sea-surface temperature [degrees C].
  real, dimension(SZIW_(ML),SZJW_(ML)) &
                                   ::  WH_UI     !< zonal ice velocities [ms-1].
  real, dimension(SZIW_(ML),SZJW_(ML)) &
                                   ::  WH_VI     !< meridional ice velocities [ms-1].
  real, dimension(SZIW_(ML),SZJW_(ML)) &
                                   ::  WH_HI     !< mean ice thickness [m].
  real, dimension(SZIW_(ML),SZJW_(ML)) &
                                   ::  WH_TS     !< ice-surface skin temperature [degrees C].
  real, dimension(SZIW_(ML),SZJW_(ML)) &
                                   ::  WH_SSS    !< sea-surface salinity [psu].
  real, dimension(SZIW_(ML),SZJW_(ML)) &
                                   ::  WH_mask   !< land-sea mask (0=land cells, 1=ocean cells)
  real, dimension(8,SZIW_(ML),SZJW_(ML)) &
                                   ::  IN_CNN    !< input variables to CNN (predict dSIC)
  real, dimension(6,SZI_(G),SZJ_(G)) &
                                   ::  IN_ANN    !< input variables to ANN (predict dCN)

  real, dimension(SZI_(G),SZJ_(G),5) &
                                   :: dCN        !< ANN predictions of category SIC corrections
  real, dimension(SZI_(G),SZJ_(G),0:5) &
                                   :: posterior  !< updated part_size (bounded between 0 and 1)

  integer :: i, j, k, m, iT, jT
  integer :: is, ie, js, je, ncat, nlay
  integer :: isdw, iedw, jsdw, jedw
  real    :: cvr, Ti, qi_new, sic_inc
  real    :: rho_ice, Cp_water
  real    :: dists, positives

  real :: hmid(5) = [0.05,0.2,0.5,0.9,2.0] !ITD thicknesses for new ice
  logical, dimension(5) :: negatives

  !parameters for adding new sea ice to a grid cell which was previously ice free
  real, parameter :: & 
       phi_init = 0.75, & !initial liquid fraction of frazil ice
       Si_new = 5.0    !salinity of mushy ice (ppt)

  !normalization statistics for both networks
  real, parameter :: &
       !CNN stats
       sic_mu = 0.2990368601723349, &
       sst_mu = 2.352753789056461, &
       ui_mu = 0.050851955768614523, &
       vi_mu = 0.016421272164475663, &
       hi_mu = 0.3624009110428704, &
       ts_mu = -4.948930143980626, &
       sss_mu = 29.957828794260223, &

       sic_std = 0.417807432477912, &
       sst_std = 5.185939952547236, &
       ui_std = 0.1232413537698019, &
       vi_std = 0.08554771683233311, &
       hi_std = 0.6317367194187057, &
       ts_std = 8.513001437482345, &
       sss_std = 10.693023255523686, &

       !ANN stats
       dsic_mu = -0.0009238007032701131, &
       cn1_mu = 0.014416831050737924, &
       cn2_mu = 0.04373226571477122, &
       cn3_mu = 0.09164522823711764, &
       cn4_mu = 0.05570272187413382, &
       cn5_mu = 0.13587840021452144, &
       
       dsic_std = 0.03622488757495426, &
       cn1_std = 0.05668652922708476, &
       cn2_std = 0.12648644666321918, &
       cn3_std = 0.2123464013448815, &
       cn4_std = 0.1633845192717932, &
       cn5_std = 0.30370125990558566

  call get_SIS2_thermo_coefs(IST%ITV, Cp_Water=Cp_water, rho_ice=rho_ice)

  is = G%isc ; ie = G%iec ; js = G%jsc ; je = G%jec ; ncat = IG%CatIce ; nlay = IG%NkIce
  isdw = ML%isdw; iedw = ML%iedw; jsdw = ML%jsdw; jedw = ML%jedw

  call pass_vector(IST%u_ice_C, IST%v_ice_C, G%Domain, stagger=CGRID_NE)
  
  !populate variables to pad for CNN halos
  WH_SIC = 0.0 ; WH_SST = 0.0 ; WH_UI = 0.0 ; WH_VI = 0.0 ; WH_HI = 0.0 ;  WH_TS = 0.0 ; WH_SSS = 0.0 ; WH_mask = 0.0
  cvr = 0.0
  do j=js,je ; do i=is,ie
     cvr = 1 - IST%part_size(i,j,0)
     WH_SIC(i,j) = cvr
     WH_SST(i,j) = OSS%SST_C(i,j)
     WH_UI(i,j) = (IST%u_ice_C(I-1,j) + IST%u_ice_C(I,j))/2
     WH_VI(i,j) = (IST%v_ice_C(i,J-1) + IST%v_ice_C(i,J))/2
     WH_TS(i,j) = FIA%Tskin_avg(i,j)
     WH_SSS(i,j) = OSS%s_surf(i,j)
     WH_mask(i,j) = G%mask2dT(i,j)
     do k=1,ncat
        WH_HI(i,j) = WH_HI(i,j) + IST%part_size(i,j,k)*(IST%mH_ice(i,j,k)/rho_ice)
     enddo
     if (cvr > 0.) then
        WH_HI(i,j) = WH_HI(i,j) / cvr
     else
        WH_HI(i,j) = 0.0
     endif
  enddo ; enddo
  
  ! Update the wide halos
  call pass_var(WH_SIC, ML%CNN_Domain)
  call pass_var(WH_SST, ML%CNN_Domain)
  call pass_vector(WH_UI, WH_VI, ML%CNN_Domain, stagger=CGRID_NE)
  call pass_var(WH_HI, ML%CNN_Domain)
  call pass_var(WH_TS, ML%CNN_Domain)
  call pass_var(WH_SSS, ML%CNN_Domain)
  call pass_var(WH_mask, ML%CNN_Domain)
  
  IN_CNN = 0.0
  ! Combine arrays for the CNN and normalize
  do j=jsdw,jedw ; do i=isdw,iedw
     if (G%mask2dT(i,j) == 1.0) then !is ocean
        IN_CNN(1,i,j) = (WH_SIC(i,j) - sic_mu)/sic_std
        IN_CNN(2,i,j) = (WH_SST(i,j) - sst_mu)/sst_std
        IN_CNN(3,i,j) = (WH_UI(i,j) - ui_mu)/ui_std
        IN_CNN(4,i,j) = (WH_VI(i,j) - vi_mu)/vi_std
        IN_CNN(5,i,j) = (WH_HI(i,j) - hi_mu)/hi_std
        IN_CNN(6,i,j) = (WH_TS(i,j) - ts_mu)/ts_std
        IN_CNN(7,i,j) = (WH_SSS(i,j) - sss_mu)/sss_std
        IN_CNN(8,i,j) = WH_mask(i,j)
     endif
  enddo; enddo

  IN_ANN = 0.0
  do j=js,je ; do i=is,ie
     if (G%mask2dT(i,j) == 1.0) then !is ocean
        IN_ANN(1,i,j) = (IST%part_size(i,j,1) - cn1_mu)/cn1_std
        IN_ANN(2,i,j) = (IST%part_size(i,j,2) - cn2_mu)/cn2_std
        IN_ANN(3,i,j) = (IST%part_size(i,j,3) - cn3_mu)/cn3_std
        IN_ANN(4,i,j) = (IST%part_size(i,j,4) - cn4_mu)/cn4_std
        IN_ANN(5,i,j) = (IST%part_size(i,j,5) - cn5_mu)/cn5_std
        IN_ANN(6,i,j) = G%mask2dT(i,j)
     endif
  enddo; enddo
  
  ! Run Python script for CNN inference
  dCN = 0.0
  call forpy_run_python(IN_CNN, IN_ANN, trim(ML%CNN_weights), trim(ML%ANN_weights), dsic_mu, dsic_std, dCN, CS)

  do j=js,je ; do i=is,ie
     do k=1,ncat
        if (G%mask2dT(i,j) == 1.0) then !is ocean
           !save predicted increment as a diagnostic
           IST%dCN(i,j,k) = dCN(i,j,k)/(432000.0/dt_slow) !Network was trained on 5-day (432000-second) increments
        endif
     enddo
  enddo; enddo

  !Update category concentrations & bound between 0 and 1
  !This part checks if the updated SIC in any category is below zero.
  !If it is, spread the equivalent negative value across the other positive categories
  !E.g if new SIC is [-0.2,0.1,0.2,0.3,0.4], then remove 0.2/4 from categories 2 through 5
  !E.g if new SIC is [-0.2,-0.1,0.4,0.2,0.1], then remove 0.3/3 from categories 3 through 5
  !This will continue in a 'while loop' until all categories are >= 0.
  posterior = 0.0
  do j=js,je ; do i=is,ie
     do k=1,ncat
        posterior(i,j,k) = IST%part_size(i,j,k) + IST%dCN(i,j,k)
     enddo
     do
        negatives = (posterior(i,j,1:) < 0.0)
        if (.not. any(negatives)) exit

        dists = 0.0
        positives = 0.0
        do k=1,ncat
           if (negatives(k)) then
              dists = dists + abs(posterior(i,j,k))
           elseif (posterior(i,j,k) > 0.0) then
              positives = positives + 1.0
           endif
        enddo

        do k=1,ncat
           if (posterior(i,j,k) > 0.0) then
              posterior(i,j,k) = posterior(i,j,k) - dists/positives
           elseif (posterior(i,j,k) < 0.0) then
              posterior(i,j,k) = 0.0
           endif   
        enddo
     enddo
     cvr = 0.0
     do k=1,ncat
        cvr = cvr + posterior(i,j,k)
     enddo
     if (cvr>1) then
        do k=1,ncat
           posterior(i,j,k) = posterior(i,j,k)/cvr
        enddo
     endif
     cvr = 0.0
     do k=1,ncat
        cvr = cvr + posterior(i,j,k)
     enddo
     posterior(i,j,0) = 1 - cvr
  enddo; enddo
  
  !update sea ice/ocean variables based on corrected sea ice state
  Ti = min(liquidus_temperature_mush(Si_new/phi_init),-0.1)
  qi_new = enthalpy_ice(Ti, Si_new)
  do j=js,je ; do i=is,ie
     cvr = 1 - posterior(i,j,0)
     sic_inc = 0.0
     do k=1,ncat
        !have added ice to grid cell which was previously ice free
        if (posterior(i,j,k)>0.0 .and. IST%part_size(i,j,k)<=0.0) then
           IST%mH_ice(i,j,k) = hmid(k)*rho_ice
           IST%mH_snow(i,j,k) = 0.0
           IST%mH_pond(i,j,k) = 0.0
           IST%enth_snow(i,j,k,1) = 0.0
           do m=1,nlay
              IST%enth_ice(i,j,k,m) = qi_new/rho_ice
              IST%sal_ice(i,j,k,m) = Si_new
           enddo
        !have removed all sea in a grid cell
        elseif (posterior(i,j,k)<=0.0 .and. IST%part_size(i,j,k)>0.0) then
           IST%mH_ice(i,j,k) = 0.0
           IST%mH_snow(i,j,k) = 0.0
           IST%mH_pond(i,j,k) = 0.0
           IST%enth_snow(i,j,k,1) = 0.0
           do m=1,nlay
              IST%enth_ice(i,j,k,m) = 0.0
              IST%sal_ice(i,j,k,m) = 0.0
           enddo
        endif
        IST%part_size(i,j,k) = posterior(i,j,k)
        sic_inc = sic_inc + IST%dCN(i,j,k)
     enddo
     IST%part_size(i,j,0) = posterior(i,j,0)
     !if (sic_inc > 0.0 .and. OSS%SST_C(i,j) > OSS%T_fr_ocn(i,j)) then
     !   IOF%flux_sh_ocn_top(i,j) = IOF%flux_sh_ocn_top(i,j) - &
     !        ((OSS%T_fr_ocn(i,j) - OSS%SST_C(i,j)) * (1035.0*Cp_water) * (ML%piston_SSTadj/86400.0)) !1035 = reference density
     !endif
 enddo; enddo

end subroutine ML_inference

! update sea ice variables as done in DA:
! /ncrc/home1/Yongfei.Zhang/dart_manhattan/models/sis/dart_to_sis.f90
  
!=======================================================================

function liquidus_temperature_mush(Sbr) result(zTin)

  ! liquidus relation: equilibrium temperature as function of brine salinity
  ! based on empirical data from Assur (1958)

  real, intent(in) :: &
       Sbr    ! ice brine salinity (ppt)

  real :: &
       zTin   ! ice layer temperature (C)

  real :: &
       t_high ! mask for high temperature liquidus region

  ! liquidus break
  real, parameter :: &
     Sb_liq =  123.66702800276086    ! salinity of liquidus break

  ! constant numbers from ice_constants.F90
  real, parameter :: &
       c1      = 1.0 , &
       c1000   = 1000

  ! liquidus relation - higher temperature region
  real, parameter :: &
       az1_liq = -18.48 ,&
       bz1_liq =   0.0
  ! liquidus relation - lower temperature region
  real, parameter :: &
       az2_liq = -10.3085,  &
       bz2_liq =  62.4

  ! basic liquidus relation constants
  real, parameter :: &
       az1p_liq = az1_liq / c1000, &
       bz1p_liq = bz1_liq / c1000, &
       az2p_liq = az2_liq / c1000, &
       bz2p_liq = bz2_liq / c1000

  ! brine salinity to temperature
  real, parameter :: &
     M1_liq = az1_liq            , &
     N1_liq = -az1p_liq          , &
     O1_liq = -bz1_liq / az1_liq , &
     M2_liq = az2_liq            , &
     N2_liq = -az2p_liq          , &
     O2_liq = -bz2_liq / az2_liq

  t_high = merge(1.0, 0.0, (Sbr <= Sb_liq))

  zTin = ((Sbr / (M1_liq + N1_liq * Sbr)) + O1_liq) * t_high + &
        ((Sbr / (M2_liq + N2_liq * Sbr)) + O2_liq) * (1.0 - t_high)

end function liquidus_temperature_mush

!=======================================================================

function enthalpy_ice(zTin, zSin) result(zqin)


  real, intent(in) :: &
       zTin, & ! ice layer temperature (C)
       zSin    ! ice layer bulk salinity (ppt)

  real :: &
       zqin    ! ice layer enthalpy (J m-3) 

  real, parameter :: CW  = 3925   ! specific heat of water ~ J/kg/K
  real, parameter :: CI  = 2100 ! specific heat of fresh ice ~ J/kg/K
  real, parameter :: LATICE  = 3.34e5   ! latent heat of fusion ~ J/kg
  real, parameter :: MIU = 0.054

  ! from cice/src/drivers/cesm/ice_constants.F90
  real :: cp_wtr, cp_ice, Lfresh, Tm
  cp_ice    = CI  ! specific heat of fresh ice (J/kg/K)
  cp_wtr    = CW   ! specific heat of ocn    (J/kg/K)
  Lfresh    = LATICE ! latent heat of melting of fresh ice (J/kg)

  Tm = - MIU*zSin

  zqin = cp_wtr*zTin + cp_ice*(zTin - Tm) + (cp_wtr - cp_ice)*Tm*log(zTin/Tm) + Lfresh*(Tm/zTin-1.0)

end function enthalpy_ice

!=======================================================================


end module SIS_ML
