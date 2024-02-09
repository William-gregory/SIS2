!> Computes a state-dependent bias correction to the part_size variable, based on a convolutional
!! neural network which has been trained to predict increments from a sea ice data assimilation
!! system. This correction is non-conservative. See Gregory et al., 2023 for details.
module SIS_G23_CNN

use ice_grid,                  only : ice_grid_type
use SIS_hor_grid,              only : SIS_hor_grid_type
use MOM_domains,               only : clone_MOM_domain,MOM_domain_type
use MOM_domains,               only : pass_var, pass_vector, CGRID_NE
use MOM_io,                    only : MOM_read_data
use MOM_EOS,                   only : EOS_type, calculate_density_derivs
use MOM_time_manager,          only : get_date, get_time, set_date, operator(-)
use SIS_diag_mediator,         only : register_SIS_diag_field
use SIS_diag_mediator,         only : post_SIS_data, post_data=>post_SIS_data
use SIS_diag_mediator,         only : SIS_diag_ctrl
use SIS_types,                 only : ice_state_type, ocean_sfc_state_type, fast_ice_avg_type, ice_ocean_flux_type
use SIS_utils,                 only : get_avg
use SIS2_ice_thm,              only : get_SIS2_thermo_coefs
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

  !CS%id_dcn = register_SIS_diag_field('ice_model', 'dCN', diag%axesTc, Time, &
  !     'ML-based correction to ice concentration', 'area fraction', missing_value=missing)

  call get_param(param_file, mdl, "CNN_HALO_SIZE", CS%CNN_halo_size, &
      "Halo size at each side of subdomains, depends on CNN architecture.", & 
      units="nondim", default=4)

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
subroutine CNN_inference(IST, OSS, FIA, IOF, G, IG, CS, US, CNN, dt_slow, Time)
  type(ice_state_type),      intent(inout)  :: IST !< A type describing the state of the sea ice
  type(fast_ice_avg_type),   intent(inout)  :: FIA !< A type containing averages of fields
                                                   !! (mostly fluxes) over the fast updates
  type(ocean_sfc_state_type), intent(inout) :: OSS !< A structure containing the arrays that describe
                                                   !! the ocean's surface state for the ice model.
  type(ice_ocean_flux_type), intent(inout) ::  IOF !< A structure containing the arrays that describe
                                                   !! the ocean's surface state for the ice model.
  type(SIS_hor_grid_type),   intent(in)     :: G      !< The horizontal grid structure
  type(ice_grid_type),       intent(in)     :: IG     !< Sea ice specific grid
  type(python_interface),    intent(in)     :: CS     !< Python interface object
  type(unit_scale_type),     intent(in)     :: US  !< A structure with unit conversion factors
  type(CNN_CS),              intent(in)     :: CNN    !< Control structure for CNN
  real,                      intent(in)     :: dt_slow !< The thermodynamic time step [T ~> s]
  type(time_type),           intent(in)     :: Time       !< The current model time. 

  !initialise input variables with wide halos
  real, dimension(SZI_(G),SZJ_(G)) &
                                   ::  HI        !< mean ice thickness [m].
  !real, dimension(SZI_(G),SZJ_(G)) &
  !                                 ::  net_sw    !< net shortwave radiation [Wm-2].
  real, dimension(SZIW_(CNN),SZJW_(CNN)) &
                                   :: WH_SIC     !< aggregate concentrations [nondim].
  real, dimension(SZIW_(CNN),SZJW_(CNN)) &
                                   :: WH_SST     !< sea-surface temperature [degrees C].
  real, dimension(SZIW_(CNN),SZJW_(CNN)) &
                                   ::  WH_UI     !< zonal ice velocities [ms-1].
  real, dimension(SZIW_(CNN),SZJW_(CNN)) &
                                   ::  WH_VI     !< meridional ice velocities [ms-1].
  real, dimension(SZIW_(CNN),SZJW_(CNN)) &
                                   ::  WH_HI     !< mean ice thickness [m].
  !real, dimension(SZIW_(CNN),SZJW_(CNN)) &
  !                                 ::  WH_SW     !< net shortwave radiation [Wm-2].
  real, dimension(SZIW_(CNN),SZJW_(CNN)) &
                                   ::  WH_TS     !< ice-surface skin temperature [degrees C].
  real, dimension(SZIW_(CNN),SZJW_(CNN)) &
                                   ::  WH_SSS    !< sea-surface salinity [ppt].
  real, dimension(SZIW_(CNN),SZJW_(CNN)) &
                                   :: WH_mask    !< land-sea mask (0=land cells, 1=ocean cells)
  real, dimension(8,SZIW_(CNN),SZJW_(CNN)) &
                                   :: XA         !< input variables to network A (predict dsiconc)
  real, dimension(6,SZI_(G),SZJ_(G)) &
                                   :: XB         !< input variables to network B (predict dCN)
  
  !initialise network outputs
  real, dimension(SZI_(G),SZJ_(G),5) &
                                   :: dCN      !< network B predictions of category SIC corrections
  !real, dimension(SZI_(G),SZJ_(G),14) &
  !                                 :: dCN      !< network B predictions of category SIC corrections
  real, dimension(SZI_(G),SZJ_(G),0:5) &
                                    :: posterior  !< updated part_size (bounded between 0 and 1)
  
  real, dimension(5) :: hmid
  integer :: b, i, j, k, m
  integer :: is, ie, js, je, ncat, nlay
  integer :: isdw, iedw, jsdw, jedw, nb
  integer :: year, month, day, hour, minute, second
  integer :: sec, yr_days
  real    :: cvr, Ti, qi_new, sw_cat, old_ice, cool_nudge
  character(85) :: filename

  type(EOS_type), pointer :: EOS => NULL()
  real :: Cp_water    ! The heat capacity of sea water [Q C-1 ~> J kg-1 degC-1]
  real :: drho_dT(1)  ! The partial derivative of density with temperature [R C-1 ~> kg m-3 degC-1]
  real :: drho_dS(1)  ! The partial derivative of density with salinity [R S-1 ~> kg m-3 ppt-1]
  real :: pres_0(1)   ! An array of pressures [Pa]
  real, parameter :: rho_ice = 905.0 ! The nominal density of sea ice [R ~> kg m-3]
  real, parameter :: nudge_sea_ice_rate = 10000.0 ! [W m-2]
  real, parameter :: &    !from ice_therm_vertical.F90
       phi_init = 0.75, & !initial liquid fraction of frazil ice
       Si_new = 5.0       !salinity of mushy ice (ppt)

  is = G%isc ; ie = G%iec ; js = G%jsc ; je = G%jec ; ncat = IG%CatIce ; nlay = IG%NkIce
  isdw = CNN%isdw; iedw = CNN%iedw; jsdw = CNN%jsdw; jedw = CNN%jedw
  nb = size(FIA%flux_sw_top,4)

  hmid = 0.0; HI = 0.0
  hmid(1) = 0.05 ; hmid(2) = 0.2 ; hmid(3) = 0.5 ; hmid(4) = 0.9 ; hmid(5) = 1.1
  do i=is,ie ; do j=js,je !compute sithick
     cvr = 1 - IST%part_size(i,j,0)
     do k=1,ncat
        HI(i,j) = HI(i,j) + IST%part_size(i,j,k)*(IST%mH_ice(i,j,k)*(US%Z_to_m/rho_ice))
     enddo
     if (cvr > 0.) then
        HI(i,j) = HI(i,j) / cvr
     else
        HI(i,j) = 0.0
     endif
  enddo; enddo

  !net_sw = 0.0
  !do j=js,je; do i=is,ie
  !    do k=0,ncat
  !       sw_cat = 0
  !       do b=1,nb
  !          sw_cat = sw_cat + FIA%flux_sw_top(i,j,k,b)
  !       enddo
  !       net_sw(i,j) = net_sw(i,j) + IST%part_size(i,j,k) * sw_cat
  !    enddo
  !enddo; enddo

  !Network does not generalize to diurnal cycle of net shortwave, so will use a daily climatology instead
  !year = 0; month = 0; day = 0; hour = 0; minute = 0; second = 0
  !sec = 0; yr_days = 0; net_sw = 0.0
  !call get_date(Time, year, month, day, hour, minute, second)
  !call get_time(Time - set_date(year, 1, 1, 0, 0, 0), sec, yr_days)
  !write(filename, "(A,I3.3,A)") "/gpfs/f5/gfdl_o/scratch/William.Gregory/CNNForpy/SWclim/1982-2017_netSWclim_day", yr_days + 1, ".nc"
  !call MOM_read_data(filename=filename, fieldname='SW', data=net_sw, MOM_Domain=G%Domain, timelevel=1, global_file=.true.)

  call pass_vector(IST%u_ice_C, IST%v_ice_C, G%Domain, stagger=CGRID_NE)
  
  !populate variables to pad for CNN halos
  WH_SIC = 0.0; WH_SST = 0.0; WH_HI = 0.0; WH_UI = 0.0; WH_VI = 0.0
  WH_TS = 0.0; WH_SSS = 0.0; WH_mask = 0.0; XB = 0.0!; WH_SW = 0.0
  do j=js,je ; do i=is,ie
     WH_SIC(i,j) = 1 - IST%part_size(i,j,0)
     WH_SST(i,j) = OSS%SST_C(i,j)
     WH_UI(i,j) = (IST%u_ice_C(I-1,j) + IST%u_ice_C(I,j))/2
     WH_VI(i,j) = (IST%v_ice_C(i,J-1) + IST%v_ice_C(i,J))/2
     WH_HI(i,j) = HI(i,j)
     !WH_SW(i,j) = net_sw(i,j)
     WH_TS(i,j) = FIA%Tskin_avg(i,j)
     WH_SSS(i,j) = OSS%s_surf(i,j)
     WH_mask(i,j) = G%mask2dT(i,j)
     do k=1,ncat
        XB(k,i,j) = IST%part_size(i,j,k)
     enddo
     XB(6,i,j) = G%mask2dT(i,j)
  enddo ; enddo
  
  ! Update the wide halos
  call pass_var(WH_SIC, CNN%CNN_Domain)
  call pass_var(WH_SST, CNN%CNN_Domain)
  call pass_vector(WH_UI, WH_VI, CNN%CNN_Domain, stagger=CGRID_NE)
  call pass_var(WH_HI, CNN%CNN_Domain)
  !call pass_var(WH_SW, CNN%CNN_Domain)
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
     !XA(6,i,j) = WH_SW(i,j)
     XA(6,i,j) = WH_TS(i,j)
     XA(7,i,j) = WH_SSS(i,j)
     XA(8,i,j) = WH_mask(i,j)
  enddo ; enddo

  ! Run Python script for CNN inference
  dCN = 0.0
  call forpy_run_python(XA, XB, dCN, CS, dt_slow)
  call pass_var(dCN, G%Domain)
  
  !do j=js,je ; do i=is,ie
  !   do k=1,ncat
  !      IST%dCN(i,j,k) = dCN(i,j,k)
  !   enddo
  !   IST%WG_SIC(i,j) = dCN(i,j,6)
  !   IST%WG_SST(i,j) = dCN(i,j,7)
  !   IST%WG_UI(i,j) = dCN(i,j,8)
  !   IST%WG_VI(i,j) = dCN(i,j,9)
  !   IST%WG_HI(i,j) = dCN(i,j,10)
  !   IST%WG_SW(i,j) = dCN(i,j,11)
  !   IST%WG_TS(i,j) = dCN(i,j,12)
  !   IST%WG_SSS(i,j) = dCN(i,j,13)
  !   IST%WG_mask(i,j) = dCN(i,j,14)
  !enddo; enddo

  !call pass_var(IST%WG_SIC, G%Domain)
  !call pass_var(IST%WG_SST, G%Domain)
  !call pass_vector(IST%WG_UI, IST%WG_VI, G%Domain, stagger=CGRID_NE)
  !call pass_var(IST%WG_HI, G%Domain)
  !call pass_var(IST%WG_SW, G%Domain)
  !call pass_var(IST%WG_TS, G%Domain)
  !call pass_var(IST%WG_SSS, G%Domain)
  !call pass_var(IST%WG_mask, G%Domain)
  !call pass_var(IST%dCN, G%Domain)

  !Update category concentrations & bound between 0 and 1
  posterior = 0.0
  do j=js,je ; do i=is,ie
     cvr = 0.0
     do k=1,ncat
        IST%dCN(i,j,k) = dCN(i,j,k) !save for diagnostic
        posterior(i,j,k) = IST%part_size(i,j,k) + IST%dCN(i,j,k) 
        if (posterior(i,j,k)<0.0) then
           posterior(i,j,k) = 0.0
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
  if (.not.allocated(IOF%melt_nudge)) allocate(IOF%melt_nudge(is:ie,js:je))
  IOF%melt_nudge(:,:) = 0.0
  pres_0(:) = 0.0
  cool_nudge = 0
  call get_SIS2_thermo_coefs(IST%ITV, Cp_Water=Cp_water)
  do j=js,je ; do i=is,ie
     cvr = 1 - posterior(i,j,0)
     old_ice = 1 - IST%part_size(i,j,0)
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
     enddo
     if (old_ice < cvr .and. cvr >=0.15 .and. OSS%SST_C(i,j) > OSS%T_fr_ocn(i,j)) then
        cool_nudge = nudge_sea_ice_rate * (cvr - old_ice)**2.0 ! W/m2
        call calculate_density_derivs(OSS%SST_C(i:i,j), OSS%s_surf(i:i,j), pres_0, &
                              drho_dT, drho_dS, 1, 1, EOS)
        IOF%melt_nudge(i,j) = (-cool_nudge*drho_dT(1)) / &
                ((Cp_water*drho_dS(1)) * max(OSS%s_surf(i,j), 1.0*US%ppt_to_S) )
     endif
  enddo; enddo
  do j=js,je ; do i=is,ie
     do k=0,ncat
        IST%part_size(i,j,k) = posterior(i,j,k)
     enddo
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
