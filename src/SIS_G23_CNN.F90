!> Computes a state-dependent bias correction to the part_size variable, based on a convolutional
!! neural network which has been trained to predict increments from a sea ice data assimilation
!! system. This correction is non-conservative. See https://doi.org/10.1029/2023MS003757 for details
module SIS_G23_CNN

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
use ftorch,                    only : torch_model, torch_tensor, torch_tensor_from_array, torch_model_forward, torch_model_load, torch_delete, torch_kCPU

implicit none; private

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

public :: CNN_init,CNN_inference

!> Control structure for CNN
type, public :: CNN_CS ; private
  type(MOM_domain_type), pointer :: CNN_Domain => NULL()  !< Domain for inputs/outputs for CNN
  integer :: isdw !< The lower i-memory limit for the wide halo arrays.
  integer :: iedw !< The upper i-memory limit for the wide halo arrays.
  integer :: jsdw !< The lower j-memory limit for the wide halo arrays.
  integer :: jedw !< The upper j-memory limit for the wide halo arrays.
  integer :: CNN_halo_size  !< Halo size at each side of subdomains
  real    :: piston_SSTadj !< piston velocity of SST restoring
  character(len=300)  :: netA_script !< NetA TorchScript
  character(len=300)  :: netB_script !< NetB TorchScript

  type(SIS_diag_ctrl), pointer :: diag => NULL() !< A type that regulates diagnostics output
  !>@{ Diagnostic handles
  integer :: id_dCN = -1
  !>@}
  
end type CNN_CS

contains

!> Prepare CNN input variables
subroutine CNN_init(Time,G,param_file,diag,CS)
  type(time_type),               intent(in)    :: Time       !< The current model time.
  type(SIS_hor_grid_type),       intent(in)    :: G     !< The horizontal grid structure.
  type(param_file_type),         intent(in)    :: param_file !< Parameter file parser structure.
  type(SIS_diag_ctrl), target,       intent(inout) :: diag  !< Diagnostics structure.
  type(CNN_CS),                  intent(inout) :: CS    !< Control structure for CNN
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

    call get_param(param_file, mdl, "NETA_SCRIPT", CS%netA_script, &
      "TorchScript of Network A with optimized weights", &
      default="/gpfs/f5/scratch/gfdl_o/William.Gregory/Ftorch/scripts/NetA_script.pt")

  call get_param(param_file, mdl, "NETB_SCRIPT", CS%netB_script, &
      "TorchScript of Network B with optimized weights", &
      default="/gpfs/f5/scratch/gfdl_o/William.Gregory/Ftorch/scripts/NetB_script.pt")
  
  wd_halos(1) = CS%CNN_halo_size
  wd_halos(2) = CS%CNN_halo_size
  if (G%symmetric) then
     call clone_MOM_domain(G%Domain, CS%CNN_Domain, min_halo=wd_halos, symmetric=.true.)
  else
     call clone_MOM_domain(G%Domain, CS%CNN_Domain, min_halo=wd_halos, symmetric=.false.)
  endif
  CS%isdw = G%isc-wd_halos(1) ; CS%iedw = G%iec+wd_halos(1)
  CS%jsdw = G%jsc-wd_halos(2) ; CS%jedw = G%jec+wd_halos(2)

end subroutine CNN_init

!> Manage input and output of CNN model
subroutine CNN_inference(IST, OSS, FIA, IOF, G, IG, CNN, dt_slow)
  type(ice_state_type),      intent(inout)  :: IST !< A type describing the state of the sea ice
  type(fast_ice_avg_type),   intent(inout)  :: FIA !< A type containing averages of fields
                                                   !! (mostly fluxes) over the fast updates
  type(ocean_sfc_state_type), intent(inout) :: OSS !< A structure containing the arrays that describe
                                                   !! the ocean's surface state for the ice model.
  type(ice_ocean_flux_type), intent(inout)  :: IOF !< A structure containing fluxes from the ice to
                                                   !! the ocean that are calculated by the ice model.
  type(SIS_hor_grid_type),   intent(in)     :: G       !< The horizontal grid structure
  type(ice_grid_type),       intent(in)     :: IG      !< Sea ice specific grid
  type(CNN_CS),              intent(in)     :: CNN     !< Control structure for CNN
  real,                      intent(in)     :: dt_slow !< The thermodynamic time step [T ~> s]

  type(torch_model) :: model_ftorch !ftorch
  real, dimension(SZIW_(CNN),SZJW_(CNN)) &
                                   ::  WH_SIC    !< aggregate concentrations [nondim].
  real, dimension(SZIW_(CNN),SZJW_(CNN)) &
                                   ::  WH_SST    !< sea-surface temperature [degrees C].
  real, dimension(SZIW_(CNN),SZJW_(CNN)) &
                                   ::  WH_UI     !< zonal ice velocities [ms-1].
  real, dimension(SZIW_(CNN),SZJW_(CNN)) &
                                   ::  WH_VI     !< meridional ice velocities [ms-1].
  real, dimension(SZIW_(CNN),SZJW_(CNN)) &
                                   ::  WH_HI     !< mean ice thickness [m].
  real, dimension(SZIW_(CNN),SZJW_(CNN)) &
                                   ::  WH_TS     !< ice-surface skin temperature [degrees C].
  real, dimension(SZIW_(CNN),SZJW_(CNN)) &
                                   ::  WH_SSS    !< sea-surface salinity [ppt].
  real, dimension(SZIW_(CNN),SZJW_(CNN)) &
                                   ::  WH_mask   !< land-sea mask (0=land cells, 1=ocean cells)
  real, dimension(7,SZIW_(CNN),SZJW_(CNN)) &
                                   ::  XA        !< input variables to network A (predict dsiconc)
  type(torch_tensor), dimension(7,SZIW_(CNN),SZJW_(CNN)) &
                                   :: XA_torch   !< input array to network A passed to PyTorch
  real, dimension(7,SZI_(G),SZJ_(G)) &
                                   ::  XB        !< input variables to network B (predict dCN)
  type(torch_tensor), dimension(7,SZI_(G),SZJ_(G)) &
                                   :: XB_torch   !< input array to network B passed to PyTorch
  
  !initialise network outputs
  real, dimension(SZI_(G),SZJ_(G)) &
                                   :: dSIC     !< network A predictions of aggregate SIC corrections
  type(torch_tensor), dimension(SZI_(G),SZJ_(G)) &
                                   :: dSIC_torch   !< network A predictions of aggregate SIC corrections
  real, dimension(5,SZI_(G),SZJ_(G)) &
                                   :: dCN      !< network B predictions of category SIC corrections
  type(torch_tensor), dimension(5,SZI_(G),SZJ_(G)) &
                                   :: dCN_torch    !< network B predictions of category SIC corrections
  
  real, dimension(SZI_(G),SZJ_(G),0:5) &
                                    :: posterior  !< updated part_size (bounded between 0 and 1)
  
  integer :: i, j, k, m
  integer :: is, ie, js, je, ncat, nlay
  integer :: isdw, iedw, jsdw, jedw
  real    :: cvr, Ti, qi_new, sic_inc
  real    :: rho_ice, Cp_water
  real    :: dists, positives

  real, dimension(5) :: hmid
  logical, dimension(5) :: negatives
  real, parameter :: & 
       phi_init = 0.75, & !initial liquid fraction of frazil ice
       Si_new = 5.0, &    !salinity of mushy ice (ppt)
       sic_mu = 0.2990368601723349, & !normalization statistics for CNNs
       sst_mu = 2.352753789056461, &
       ui_mu = 0.050851955768614523, &
       vi_mu = 0.016421272164475663, &
       hi_mu = 0.3624009110428704, &
       ts_mu = -4.948930143980626, &
       sss_mu = 29.957828794260223, &
       dsic_mu = -0.0009238007032701131, &
       cn1_mu = 0.014416831050737924, &
       cn2_mu = 0.04373226571477122, &
       cn3_mu = 0.09164522823711764, &
       cn4_mu = 0.05570272187413382, &
       cn5_mu = 0.13587840021452144, &
       sic_std = 0.417807432477912, &
       sst_std = 5.185939952547236, &
       ui_std = 0.1232413537698019, &
       vi_std = 0.08554771683233311, &
       hi_std = 0.6317367194187057, &
       ts_std = 8.513001437482345, &
       sss_std = 10.693023255523686, &
       dsic_std = 0.03622488757495426, &
       cn1_std = 0.05668652922708476, &
       cn2_std = 0.12648644666321918, &
       cn3_std = 0.2123464013448815, &
       cn4_std = 0.1633845192717932, &
       cn5_std = 0.30370125990558566

  call get_SIS2_thermo_coefs(IST%ITV, Cp_Water=Cp_water, rho_ice=rho_ice)

  is = G%isc ; ie = G%iec ; js = G%jsc ; je = G%jec ; ncat = IG%CatIce ; nlay = IG%NkIce
  isdw = CNN%isdw; iedw = CNN%iedw; jsdw = CNN%jsdw; jedw = CNN%jedw
  
  !ITD thicknesses for new ice
  hmid(1) = 0.05 ; hmid(2) = 0.2 ; hmid(3) = 0.5 ; hmid(4) = 0.9 ; hmid(5) = 2.0

  call pass_vector(IST%u_ice_C, IST%v_ice_C, G%Domain, stagger=CGRID_NE)
  
  !populate variables to pad for CNN halos
  WH_SIC = 0.0; WH_SST = 0.0; WH_HI = 0.0; WH_UI = 0.0; WH_VI = 0.0
  WH_TS = 0.0; WH_SSS = 0.0; WH_mask = 0.0; XB = 0.0
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
        XB(k+1,i,j) = IST%part_size(i,j,k)
        WH_HI(i,j) = WH_HI(i,j) + IST%part_size(i,j,k)*(IST%mH_ice(i,j,k)/rho_ice)
     enddo
     XB(7,i,j) = G%mask2dT(i,j)
     if (cvr > 0.) then
        WH_HI(i,j) = WH_HI(i,j) / cvr
     else
        WH_HI(i,j) = 0.0
     endif
  enddo ; enddo
  
  ! Update the wide halos
  call pass_var(WH_SIC, CNN%CNN_Domain)
  call pass_var(WH_SST, CNN%CNN_Domain)
  call pass_vector(WH_UI, WH_VI, CNN%CNN_Domain, stagger=CGRID_NE)
  call pass_var(WH_HI, CNN%CNN_Domain)
  call pass_var(WH_TS, CNN%CNN_Domain)
  call pass_var(WH_SSS, CNN%CNN_Domain)
  call pass_var(WH_mask, CNN%CNN_Domain)

  ! Combine arrays for CNN input
  XA = 0.0
  do j=jsdw,jedw ; do i=isdw,iedw
     if (G%mask2dT(i,j) == 0.0) then !set land values to zero
        XA(1,i,j) = 0.0
        XA(2,i,j) = 0.0
        XA(3,i,j) = 0.0
        XA(4,i,j) = 0.0
        XA(5,i,j) = 0.0
        XA(6,i,j) = 0.0
        !XA(7,i,j) = 0.0
     else
        XA(1,i,j) = (WH_SIC(i,j) - sic_mu)/sic_std
        XA(2,i,j) = (WH_SST(i,j) - sst_mu)/sst_std
        XA(3,i,j) = (WH_UI(i,j) - ui_mu)/ui_std
        XA(4,i,j) = (WH_VI(i,j) - vi_mu)/vi_std
        XA(5,i,j) = (WH_HI(i,j) - hi_mu)/hi_std
        XA(6,i,j) = (WH_TS(i,j) - ts_mu)/ts_std
        !XA(7,i,j) = (WH_SSS(i,j) - sss_mu)/sss_std
        XA(7,i,j) = WH_mask(i,j)
     endif
  enddo ; enddo

  !Load PyTorch model for dSIC predictions
  dSIC = 0.0
  call torch_model_load(model_ftorch, CNN%netA_script)
  call torch_tensor_from_array(XA_torch, XA, [1,2,3], torch_kCPU)
  call torch_tensor_from_array(dSIC_torch, dSIC, [1,2], torch_kCPU)
  call torch_model_forward(model_ftorch, XA_torch, dSIC_torch)

  !need to handle squeezing of outputs!!
  do j=js,je ; do i=is,ie
     if (G%mask2dT(i,j) == 0.0) then !set land values to zero
        XB(1,i,j) = 0.0
        XB(2,i,j) = 0.0
        XB(3,i,j) = 0.0
        XB(4,i,j) = 0.0
        XB(5,i,j) = 0.0
        XB(6,i,j) = 0.0
     else   
        XB(1,i,j) = (dSIC(i,j) - dsic_mu)/dsic_std
        XB(2,i,j) = (XB(2,i,j) - cn1_mu)/cn1_std
        XB(3,i,j) = (XB(3,i,j) - cn2_mu)/cn2_std
        XB(4,i,j) = (XB(4,i,j) - cn3_mu)/cn3_std
        XB(5,i,j) = (XB(5,i,j) - cn4_mu)/cn4_std
        XB(6,i,j) = (XB(6,i,j) - cn5_mu)/cn5_std
     endif
  enddo ; enddo

  !Load PyTorch model for dCN predictions
  dCN = 0.0
  call torch_model_load(model_ftorch, CNN%netB_script)
  call torch_tensor_from_array(XB_torch, XB, [1,2,3], torch_kCPU)
  call torch_tensor_from_array(dCN_torch, dCN, [1,2,3], torch_kCPU)
  call torch_model_forward(model_ftorch, XB_torch, dCN_torch)

  call torch_delete(XA_torch)
  call torch_delete(XB_torch)
  call torch_delete(dSIC_torch)
  call torch_delete(dCN_torch)
  do j=js,je ; do i=is,ie
     if (G%mask2dT(i,j) == 0.0) then !is land
        do k=1,ncat
           dCN(k,i,j) = 0
        enddo
     endif
  enddo; enddo
  call pass_var(dCN, G%Domain)

  !Update category concentrations & bound between 0 and 1
  !This part checks if the updated SIC in any category is below zero.
  !If it is, spread the equivalent negative value across the other positive categories
  !E.g if new SIC is [-0.2,0.1,0.2,0.3,0.4], then remove 0.2/4 from categories 2 through 5
  !E.g if new SIC is [-0.2,-0.1,0.4,0.2,0.1], then remove 0.3/3 from categories 3 through 5
  !This will continue in a 'while loop' until all categories are >= 0.
  posterior = 0.0
  do j=js,je ; do i=is,ie
     do k=1,ncat
        IST%dCN(i,j,k) = dCN(k,i,j)/(432000.0/dt_slow)
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
     if (sic_inc > 0.0 .and. OSS%SST_C(i,j) > OSS%T_fr_ocn(i,j)) then
        IOF%flux_sh_ocn_top(i,j) = IOF%flux_sh_ocn_top(i,j) - &
             ((OSS%T_fr_ocn(i,j) - OSS%SST_C(i,j)) * (1035.0*Cp_water) * (CNN%piston_SSTadj/86400.0)) !1035 = reference density
     endif
  endif
 enddo; enddo
     
end subroutine CNN_inference

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


end module SIS_G23_CNN
