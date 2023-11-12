!> Computes a state-dependent bias correction to the part_size variable, based on a convolutional
!! neural network which has been trained to predict increments from a sea ice data assimilation
!! system. This correction is non-conservative. See Gregory et al., 2023 for details.
module SIS_G23_CNN

use ice_grid,                  only : ice_grid_type
use SIS_hor_grid,              only : SIS_hor_grid_type
use MOM_domains,               only : clone_MOM_domain,MOM_domain_type
use MOM_domains,               only : pass_var
use SIS_diag_mediator,         only : register_SIS_diag_field
use SIS_diag_mediator,         only : post_SIS_data, post_data=>post_SIS_data
use SIS_diag_mediator,         only : SIS_diag_ctrl
use SIS_types,                 only : ice_state_type, ocean_sfc_state_type, fast_ice_avg_type
use SIS_utils,                 only : get_avg
use MOM_diag_mediator,         only : time_type
use MOM_unit_scaling,          only : unit_scale_type
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

  type(SIS_diag_ctrl), pointer :: diag => NULL() !< A type that regulates diagnostics output
  !>@{ Diagnostic handles
  integer :: id_dSICN = -1
  !>@}
  
end type CNN_CS

contains

!> Prepare CNN input variables
subroutine CNN_init(Time,G,US,param_file,diag,CS)
  type(time_type),               intent(in)    :: Time       !< The current model time.
  type(SIS_hor_grid_type),       intent(in)    :: G     !< The horizontal grid structure.
  type(unit_scale_type),         intent(in)    :: US         !< A dimensional unit scaling type
  type(param_file_type),         intent(in)    :: param_file !< Parameter file parser structure.
  type(SIS_diag_ctrl), target,       intent(inout) :: diag  !< Diagnostics structure.
  type(CNN_CS),                  intent(inout) :: CS    !< Control structure for CNN
  ! Local Variables
  !integer :: wd_halos(2) ! Varies with CNN
  real, parameter :: missing = -1e34
  character(len=40)  :: mdl = "SIS_CNN"  ! module name

  ! Register fields for output from this module.
  CS%diag => diag

  CS%id_dSICN = register_SIS_diag_field('ice_model', 'dSICN', diag%axesTc, Time, &
       'ML-based correction to ice concentration', 'area fraction', missing_value=missing)

  call get_param(param_file, mdl, "CNN_HALO_SIZE", CS%CNN_halo_size, &
      "Halo size at each side of subdomains, depends on CNN architecture.", & 
      units="nondim", default=4)

  !wd_halos(1) = CS%CNN_halo_size
  !wd_halos(2) = CS%CNN_halo_size
  !call clone_MOM_domain(G%Domain, CS%CNN_Domain, min_halo=wd_halos, symmetric=.true.)
  !CS%isdw = G%isc-wd_halos(1) ; CS%iedw = G%iec+wd_halos(1)
  !CS%jsdw = G%jsc-wd_halos(2) ; CS%jedw = G%jec+wd_halos(2)

end subroutine CNN_init

!> Manage input and output of CNN model
subroutine CNN_inference(IST, OSS, FIA, G, IG, CS, CNN, dt_slow)
  type(ice_state_type),      intent(inout)  :: IST !< A type describing the state of the sea ice
  type(fast_ice_avg_type),   intent(in)     :: FIA !< A type containing averages of fields
                                                   !! (mostly fluxes) over the fast updates
  type(ocean_sfc_state_type), intent(inout) :: OSS !< A structure containing the arrays that describe
                                                   !! the ocean's surface state for the ice model.
  type(SIS_hor_grid_type),   intent(in)     :: G      !< The horizontal grid structure
  type(ice_grid_type),       intent(in)     :: IG     !< Sea ice specific grid
  type(python_interface),    intent(in)     :: CS     !< Python interface object
  type(CNN_CS),              intent(in)     :: CNN    !< Control structure for CNN
  real,                      intent(in)     :: dt_slow !< The thermodynamic time step [T ~> s]

  !initialise input variables with wide halos
  real, dimension(SZI_(G),SZJ_(G)) &
                                   ::  HI        !< mean ice thickness [m].
  real, dimension(SZI_(G),SZJ_(G)) &
                                   ::  net_sw    !< net shortwave radiation [Wm-2].
  real, dimension(9,SZI_(G),SZJ_(G)) &
                                   :: XA         !< input variables to network A (predict dSIC)
  real, dimension(6,SZI_(G),SZJ_(G)) &
                                   :: XB         !< input variables to network B (predict dSICN)
  
  !initialise network outputs
  real, dimension(SZI_(G),SZJ_(G),5) &
                                   :: dSICN      !< network B predictions of category SIC corrections
  real, dimension(SZI_(G),SZJ_(G),0:5) &
                                   :: posterior  !< updated part_size (bounded between 0 and 1)
  real, dimension(5) :: hmid
  integer :: b, i, j, k, m
  integer :: is, ie, js, je, ncat, nlay, nb
  real    :: cvr, Ti, qi_new, sw_cat
  real, parameter :: rho_ice = 905.0
  real, parameter :: &    !from ice_therm_vertical.F90
       phi_init = 0.75, & !initial liquid fraction of frazil ice
       Si_new = 5.0       !salinity of mushy ice

  is = G%isc ; ie = G%iec ; js = G%jsc ; je = G%jec ; ncat = IG%CatIce ; nlay = IG%NkIce
  nb = size(FIA%flux_sw_top,4)

  hmid = 0.0; HI = 0.0
  hmid(1) = 0.05 ; hmid(2) = 0.2 ; hmid(3) = 0.5 ; hmid(4) = 0.9 ; hmid(5) = 1.1
  call get_avg(IST%mH_ice, IST%part_size(:,:,1:), HI, wtd=.true.) ! compute sithick

  do j=js,je !compute net shortwave
     do i=is,ie ; net_sw(i,j) = 0.0 ; enddo
     do k=0,ncat ; do i=is,ie
        sw_cat = 0 ; do b=1,nb ; sw_cat = sw_cat + FIA%flux_sw_top(i,j,k,b) ; enddo
        net_sw(i,j) = net_sw(i,j) + IST%part_size(i,j,k) * sw_cat
     enddo; enddo
  enddo

  !populate variables
  XA = 0.0 ; XB = 0.0
  do j=js,je ; do i=is,ie
     cvr = 0.0
     do k=1,ncat
        cvr = cvr + IST%part_size(i,j,k)
        XB(k,i,j) = IST%part_size(i,j,k)
     enddo
     XA(1,i,j) = cvr
     XA(2,i,j) = OSS%SST_C(i,j)
     XA(3,i,j) = 0.5*( IST%u_ice_C(I-1,j) + IST%u_ice_C(I,j) ) ! Copy the computational section from UI into cell center
     XA(4,i,j) = 0.5*( IST%v_ice_C(i,J-1) + IST%v_ice_C(i,J) ) ! Copy the computational section from VI into cell center
     XA(5,i,j) = HI(i,j)
     XA(6,i,j) = net_sw(i,j)
     XA(7,i,j) = FIA%Tskin_avg(i,j)
     XA(8,i,j) = OSS%s_surf(i,j)
     XA(9,i,j) = G%mask2dT(i,j)
     XB(6,i,j) = G%mask2dT(i,j)
  enddo ; enddo

  ! Run Python script for CNN inference
  dSICN = 0.0
  call forpy_run_python(XA, XB, dSICN, CS, dt_slow)

  !call pass_var(dSICN, G%Domain)
  if (CNN%id_dSICN>0)  call post_data(CNN%id_dSICN, dSICN, CNN%diag)

  !Update category concentrations & bound between 0 and 1
  posterior = 0.0
  do j=js,je ; do i=is,ie
     cvr = 0.0
     do k=1,ncat
        posterior(i,j,k) = IST%part_size(i,j,k) + dSICN(i,j,k) 
        if (posterior(i,j,k)<0) then
           posterior(i,j,k) = 0
        endif
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
     cvr = 0.0
     do k=1,ncat
        !have added ice to grid cell which was previously ice free
        if (posterior(i,j,k)>0 .and. IST%part_size(i,j,k)<=0) then
           IST%mH_ice(i,j,k) = hmid(k)*rho_ice
           IST%mH_snow(i,j,k) = 0
           IST%enth_snow(i,j,k,1) = 0
           IST%T_surf(i,j,k) = Ti
           IST%mH_pond(i,j,k) = 0
        do m=1,nlay
           IST%enth_ice(i,j,k,m) = qi_new
           IST%sal_ice(i,j,k,m) = Si_new
        enddo
        !have removed all sea in a grid cell
        elseif (posterior(i,j,k)<=0 .and. IST%part_size(i,j,k)>0) then
           IST%mH_ice(i,j,k) = 0
           IST%mH_snow(i,j,k) = 0
           IST%enth_snow(i,j,k,1) = 0
           IST%T_surf(i,j,k) = OSS%T_fr_ocn(i,j) !freezing point based on salinity
           IST%mH_pond(i,j,k) = 0
           do m=1,nlay
              IST%enth_ice(i,j,k,m) = 0
              IST%sal_ice(i,j,k,m) = 0
           enddo
        endif
        cvr = cvr + posterior(i,j,k)
     enddo
     !if (cvr>=0.3) then
     !   OSS%SST_C(i,j) = OSS%T_fr_ocn(i,j) !adjust SST under sea ice to freezing point
     !endif
     !if (OSS%SST_C(i,j)<-2) then
     !   OSS%SST_C(i,j) = -2
     !endif
  enddo; enddo
  do j=js,je ; do i=is,ie
     do k=0,ncat
        IST%part_size(i,j,k) = posterior(i,j,k)
     enddo
  enddo; enddo
     
end subroutine CNN_inference

! update sea ice variables as done in DA:
! /ncrc/home1/Yongfei.Zhang/dart_manhattan/models/sis/dart_to_sis.f90
  
function enthalpy_ice(zTin, zSin) result(zqin)

  real, intent(in) :: &
       zTin, & !ice layer temperature (C)
       zSin    !ice layer bulk salinity (ppt)

  real, parameter :: cp_wtr = 4200   !specific heat capacity of water ~ J/kg/K
  real, parameter :: cp_ice = 2100   !specific heat capacity of ice ~ J/kg/K
  real, parameter :: Lfresh = 3.34e5 !latent heat of fusion ~ J/kg
  real, parameter :: MIU = 0.054
  real :: &
       zqin, &    ! ice layer enthalpy (J m-3)
       Tm

  Tm = - MIU*zSin
  zqin = cp_wtr*zTin + cp_ice*(zTin - Tm) + (cp_wtr - cp_ice)*Tm*log(zTin/Tm) + Lfresh*(Tm/zTin-1)

end function enthalpy_ice

!=======================================================================

function liquidus_temperature_mush(Sbr) result(Ti)

  real, intent(in) :: &
       Sbr
  real, parameter :: Sb_liq = 123.66702800276086
  real, parameter :: c1 = 1.0
  real, parameter :: c1000 = 1000
  real, parameter :: az1_liq = -18.48
  real, parameter :: bz1_liq = 0.0
  real, parameter :: az2_liq = -10.3085
  real, parameter :: bz2_liq = 62.4
  real :: az1p_liq, bz1p_liq, az2p_liq, bz2p_liq, O1_liq, O2_liq, t_high
  real :: Ti

  if (Sbr<=Sb_liq) then
     t_high = 1.0
  else
     t_high = 0
  endif

  az1p_liq = az1_liq / c1000
  bz1p_liq = bz1_liq / c1000
  az2p_liq = az2_liq / c1000
  bz2p_liq = bz2_liq / c1000
  O1_liq = -bz1_liq / az1_liq
  O2_liq = -bz2_liq / az2_liq

  Ti = ((Sbr / (az1_liq - az1p_liq * Sbr)) + O1_liq) * t_high + ((Sbr / (az2_liq - az2p_liq * Sbr)) + O2_liq) * (1.0 - t_high)

end function liquidus_temperature_mush

!=======================================================================

end module SIS_G23_CNN
