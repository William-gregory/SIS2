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
use SIS_types,                 only : ice_state_type, ocean_sfc_state_type, fast_ice_avg_type, ice_ocean_flux_type
use SIS2_ice_thm,              only : get_SIS2_thermo_coefs, enthalpy_liquid_freeze
use MOM_diag_mediator,         only : time_type
use MOM_file_parser,           only : get_param, param_file_type
use Forpy_interface,           only : forpy_run_python, python_interface

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
  logical :: do_SSTadj !< apply a heat flux under sea ice
  real    :: piston_SSTadj !< piston velocity of SST restoring
  character(len=300)  :: netA_weights !< Optimized weights for Network A
  character(len=300)  :: netB_weights
  character(len=300)  :: netA_stats
  character(len=300)  :: netB_stats

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

  call get_param(param_file, mdl, "DO_SSTADJ", CS%do_SSTadj, &
      "Whether to apply heat flux under sea ice for sea ice added by CNN", & 
      default=.false.)

  call get_param(param_file, mdl, "PISTON_SSTADJ", CS%piston_SSTadj, &
      "Piston velocity with which to restore SST after CNN correction", &
      units="m day-1", default=4.0)

  call get_param(param_file, mdl, "NETA_WEIGHTS", CS%netA_weights, &
      "Optimized weights for Network A", &
      default="/gpfs/f5/scratch/gfdl_o/William.Gregory/CNNForpy/NetworkA_weights_SPEAR.pt")

  call get_param(param_file, mdl, "NETB_WEIGHTS", CS%netB_weights, &
      "Optimized weights for Network B", &
      default="/gpfs/f5/scratch/gfdl_o/William.Gregory/CNNForpy/NetworkA_weights_SPEAR.pt")

  call get_param(param_file, mdl, "NETA_STATS", CS%netA_stats, &
      "Normalization statistics for Network A", &
      default="/gpfs/f5/scratch/gfdl_o/William.Gregory/CNNForpy/NetworkA_statistics_SPEAR_1982-2017.npz")

  call get_param(param_file, mdl, "NETB_STATS", CS%netB_stats, &
      "Normalization statistics for Network B", &
      default="/gpfs/f5/scratch/gfdl_o/William.Gregory/CNNForpy/NetworkB_statistics_SPEAR_1982-2017.npz")
  
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
subroutine CNN_inference(IST, OSS, FIA, IOF, G, IG, CS, CNN, dt_slow)
  type(ice_state_type),      intent(inout)  :: IST !< A type describing the state of the sea ice
  type(fast_ice_avg_type),   intent(inout)  :: FIA !< A type containing averages of fields
                                                   !! (mostly fluxes) over the fast updates
  type(ocean_sfc_state_type), intent(inout) :: OSS !< A structure containing the arrays that describe
                                                   !! the ocean's surface state for the ice model.
  type(ice_ocean_flux_type), intent(inout)  :: IOF !< A structure containing fluxes from the ice to
                                                   !! the ocean that are calculated by the ice model.
  type(SIS_hor_grid_type),   intent(in)     :: G      !< The horizontal grid structure
  type(ice_grid_type),       intent(in)     :: IG     !< Sea ice specific grid
  type(python_interface),    intent(in)     :: CS     !< Python interface object
  type(CNN_CS),              intent(in)     :: CNN    !< Control structure for CNN
  real,                      intent(in)     :: dt_slow !< The thermodynamic time step [T ~> s]

  !initialise input variables with wide halos
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
  real, dimension(8,SZIW_(CNN),SZJW_(CNN)) &
                                   ::  XA        !< input variables to network A (predict dsiconc)
  real, dimension(6,SZI_(G),SZJ_(G)) &
                                   ::  XB        !< input variables to network B (predict dCN)
  
  !initialise network outputs
  real, dimension(SZI_(G),SZJ_(G),5) &
                                   :: dCN      !< network B predictions of category SIC corrections
  real, dimension(SZI_(G),SZJ_(G),0:5) &
                                    :: posterior  !< updated part_size (bounded between 0 and 1)
  
  real, dimension(5) :: hmid
  logical, dimension(5) :: negatives
  real    :: positives, dists
  integer :: i, j, k, m
  integer :: is, ie, js, je, ncat, nlay
  integer :: isdw, iedw, jsdw, jedw
  real    :: cvr, sithick, cvr_old

  real, dimension(IG%NkIce) :: S_col      ! The salinity of a column of ice [gSalt kg-1].
  real :: qflx_res_ice
  real :: e2m_tot     ! The total enthalpy required to melt all ice and snow [J m-2].
  real :: rho_ice
  real :: enth_units
  real :: LatHtFus
  real :: LatHtVap
  logical :: spec_thermo_sal  ! If true, use the specified salinities of the
                              ! various sub-layers of the ice for all thermodynamic
                              ! calculations; otherwise use the prognostic
                              ! salinity fields for these calculations.
  
  is = G%isc ; ie = G%iec ; js = G%jsc ; je = G%jec ; ncat = IG%CatIce ; nlay = IG%NkIce
  isdw = CNN%isdw; iedw = CNN%iedw; jsdw = CNN%jsdw; jedw = CNN%jedw

  hmid = 0.0
  hmid(1) = 0.05 ; hmid(2) = 0.2 ; hmid(3) = 0.5 ; hmid(4) = 0.9 ; hmid(5) = 1.1

  call get_SIS2_thermo_coefs(IST%ITV, ice_salinity=S_col, enthalpy_units=enth_units, &
                   rho_ice=rho_ice, specified_thermo_salinity=spec_thermo_sal, &
                   Latent_fusion=LatHtFus, Latent_vapor=LatHtVap)

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
        XB(k,i,j) = IST%part_size(i,j,k)
        WH_HI(i,j) = WH_HI(i,j) + IST%part_size(i,j,k)*(IST%mH_ice(i,j,k)/rho_ice)
     enddo
     XB(6,i,j) = G%mask2dT(i,j)
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
     XA(1,i,j) = WH_SIC(i,j)
     XA(2,i,j) = WH_SST(i,j)
     XA(3,i,j) = WH_UI(i,j)
     XA(4,i,j) = WH_VI(i,j)
     XA(5,i,j) = WH_HI(i,j)
     XA(6,i,j) = WH_TS(i,j)
     XA(7,i,j) = WH_SSS(i,j)
     XA(8,i,j) = WH_mask(i,j)
  enddo ; enddo

  ! Run Python script for CNN inference
  dCN = 0.0
  call forpy_run_python(XA, XB, CNN%netA_weights, CNN%netB_weights, CNN%netA_stats, CNN%netB_stats, dCN, CS, dt_slow)
  call pass_var(dCN, G%Domain)

  !Update category concentrations & bound between 0 and 1
  posterior = 0.0
  do j=js,je ; do i=is,ie
     do k=1,ncat
        IST%dCN(i,j,k) = dCN(i,j,k) !save for diagnostic
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

  do j=js,je ; do i=is,ie
     qflx_res_ice = 0.0
     e2m_tot = 0.0
     cvr = 1 - posterior(i,j,0)
     cvr_old = 1 - IST%part_size(i,j,0)
     sithick = 0.0
     do k=1,ncat
        !the enthalpy_liquid_freeze term is for the brine pockets in sea ice. So its enth_brine - enth_ice
        e2m_tot = (IST%part_size(i,j,k)*IST%mH_snow(i,j,k)) * IG%H_to_kg_m2 * &
                       ((enthalpy_liquid_freeze(0.0, IST%ITV) - &
                       IST%enth_snow(i,j,k,1)) / enth_units)
        if (spec_thermo_sal) then 
           do m=1,nlay
              e2m_tot = e2m_tot + (IST%part_size(i,j,k)*IST%mH_ice(i,j,k) * IG%H_to_kg_m2) * &
                           ((enthalpy_liquid_freeze(S_col(m), IST%ITV) - &
                           IST%enth_ice(i,j,k,m)) / enth_units)
           enddo
        else
           do m=1,nlay
              e2m_tot = e2m_tot + (IST%part_size(i,j,k)*IST%mH_ice(i,j,k) * IG%H_to_kg_m2) * &
                       ((enthalpy_liquid_freeze(IST%sal_ice(i,j,k,m), IST%ITV) - &
                       IST%enth_ice(i,j,k,m)) / enth_units)
           enddo
        endif
        if (posterior(i,j,k)>0 .and. IST%part_size(i,j,k)<=0) then
           sithick = sithick + hmid(k)*posterior(i,j,k)
        else
           sithick = sithick + posterior(i,j,k)*(IST%mH_ice(i,j,k)/rho_ice)
        endif   
     enddo
     if (cvr > 0.0) then
        sithick = sithick/cvr
     else
        sithick = 0.0
     endif
     
     qflx_res_ice = (LatHtFus*rho_ice*sithick*cvr-e2m_tot) / (86400.0*5.0) !5-day restoring time-scale
     if (cvr > cvr_old) then
        FIA%frazil_left(i,j) = FIA%frazil_left(i,j) + abs(qflx_res_ice)*dt_slow
     elseif (cvr < cvr_old) then
        do k=1,ncat
           FIA%bmelt(i,j,k) = FIA%bmelt(i,j,k) + abs(qflx_res_ice)*dt_slow
        enddo
     endif
     
  enddo ; enddo

end subroutine CNN_inference

!=======================================================================

end module SIS_G23_CNN
