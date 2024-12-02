!> Computes a state-dependent bias correction to the part_size variable, based on a convolutional
!! neural network which has been trained to predict increments from a sea ice data assimilation
!! system. This correction is non-conservative. See https://doi.org/10.1029/2023MS003757 for details
module SIS_G23_CNN

use ice_grid,                  only : ice_grid_type
use SIS_hor_grid,              only : SIS_hor_grid_type
use MOM_io,                    only : MOM_read_data
use MOM_domains,               only : clone_MOM_domain,MOM_domain_type
use MOM_domains,               only : pass_var, pass_vector, CGRID_NE
use SIS_diag_mediator,         only : SIS_diag_ctrl
use SIS2_ice_thm,              only : get_SIS2_thermo_coefs
use SIS_utils,                 only : is_NaN
use SIS_types,                 only : ice_state_type, ocean_sfc_state_type, fast_ice_avg_type, ice_ocean_flux_type
use MOM_diag_mediator,         only : time_type
use MOM_file_parser,           only : get_param, param_file_type

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
  real, dimension(2304)  :: netA_weight_vec1 !< 3 x 3 x 8 x 32
  real, dimension(18432) :: netA_weight_vec2 !< 3 x 3 x 32 x 64
  real, dimension(73728) :: netA_weight_vec3 !< 3 x 3 x 64 x 128
  real, dimension(1152)  :: netA_weight_vec4 !< 3 x 3 x 128 x 1
  real, dimension(224)   :: netB_weight_vec1 !< 1 x 1 x 7 x 32
  real, dimension(2048)  :: netB_weight_vec2 !< 1 x 1 x 32 x 64
  real, dimension(8192)  :: netB_weight_vec3 !< 1 x 1 x 64 x 128
  real, dimension(640)   :: netB_weight_vec4 !< 1 x 1 x 128 x 5

  character(len=102)  :: netA_weights !< filename of CNN weights
  character(len=102)  :: netB_weights !< filename of ANN weights
  
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

  call get_param(param_file, mdl, "NETA_WEIGHTS", CS%netA_weights, &
      "Network A (CNN) optimized weights", &
      default="/gpfs/f5/scratch/gfdl_o/William.Gregory/FTorch/weights/NetworkA_weights.nc")

  call get_param(param_file, mdl, "NETB_WEIGHTS", CS%netB_weights, &
      "Network B (ANN) optimized weights", &
      default="/gpfs/f5/scratch/gfdl_o/William.Gregory/FTorch/weights/NetworkB_weights.nc")

  call MOM_read_data(filename=CS%netA_weights, fieldname="C1", data=CS%netA_weight_vec1)
  call MOM_read_data(filename=CS%netA_weights, fieldname="C2", data=CS%netA_weight_vec2)
  call MOM_read_data(filename=CS%netA_weights, fieldname="C3", data=CS%netA_weight_vec3)
  call MOM_read_data(filename=CS%netA_weights, fieldname="C4", data=CS%netA_weight_vec4)
  call MOM_read_data(filename=CS%netB_weights, fieldname="C1", data=CS%netB_weight_vec1)
  call MOM_read_data(filename=CS%netB_weights, fieldname="C2", data=CS%netB_weight_vec2)
  call MOM_read_data(filename=CS%netB_weights, fieldname="C3", data=CS%netB_weight_vec3)
  call MOM_read_data(filename=CS%netB_weights, fieldname="C4", data=CS%netB_weight_vec4)
  
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

subroutine CNN_forward(IN, OUT, weights1, weights2, weights3, weights4, CNN)
  real, dimension(:,:,:), intent(in) :: IN
  real, dimension(:,:,:), intent(inout) :: OUT
  real, dimension(:), intent(in) :: weights1
  real, dimension(:), intent(in) :: weights2
  real, dimension(:), intent(in) :: weights3
  real, dimension(:), intent(in) :: weights4
  type(CNN_CS), intent(in) :: CNN
  real, dimension(:,:,:), allocatable :: tmp1
  real, dimension(:,:,:), allocatable :: tmp2
  real, dimension(:,:,:), allocatable :: tmp3
  
  integer :: i, j, x, y, z, u, v, isdw, iedw, jsdw, jedw
  integer :: x1, x2, x3, y1, y2, y3
  
  isdw = CNN%isdw; iedw = CNN%iedw; jsdw = CNN%jsdw; jedw = CNN%jedw
  x1 = SIZE(IN,2) - 2 ; y1 = SIZE(IN,3) - 2
  x2 = SIZE(IN,2) - 4 ; y2 = SIZE(IN,3) - 4
  x3 = SIZE(IN,2) - 6 ; y3 = SIZE(IN,3) - 6
  allocate(tmp1(32,x1,y1))
  allocate(tmp2(64,x2,y2))
  allocate(tmp3(128,x3,y3))
  
  OUT = 0.0
  tmp1 = 0.0
  tmp2 = 0.0
  tmp3 = 0.0
  do j=jsdw+1,jedw-1 ; do i=isdw+1,iedw-1
     z = 1
     do x=1,SIZE(IN,1)
        do y=1,32
           do u=-1,1
              do v=-1,1 
                 tmp1(y,i-1,j-1) = tmp1(y,i-1,j-1) + IN(x,i+u,j+v)*weights1(z)
                 z = z + 1
              enddo
           enddo
        enddo
     enddo
  enddo; enddo
  do j=jsdw+2,jedw-2 ; do i=isdw+2,iedw-2
     z = 1
     do x=1,32
        do y=1,64
           do u=-1,1
              do v=-1,1
                 tmp2(y,i-2,j-2) = tmp2(y,i-2,j-2) + max(0.0,tmp1(x,i-1+u,j-1+v))*weights2(z)
                 z = z + 1
              enddo
           enddo
        enddo
     enddo
  enddo; enddo
  do j=jsdw+3,jedw-3 ; do i=isdw+3,iedw-3
     z = 1
     do x=1,64
        do y=1,128
           do u=-1,1
              do v=-1,1
                 tmp3(y,i-3,j-3) = tmp3(y,i-3,j-3) + max(0.0,tmp2(x,i-2+u,j-2+v))*weights3(z)
                 z = z + 1
              enddo
           enddo
        enddo
     enddo
  enddo; enddo
  do j=jsdw+4,jedw-4 ; do i=isdw+4,iedw-4
     z = 1
     do x=1,128
        do y=1,SIZE(OUT,1)
           do u=-1,1
              do v=-1,1
                 OUT(y,i-4,j-4) = OUT(y,i-4,j-4) + max(0.0,tmp3(x,i-3+u,j-3+v))*weights4(z)
                 z = z + 1
              enddo
           enddo
        enddo
     enddo
  enddo; enddo

  do j=jsdw+4,jedw-4 ; do i=isdw+4,iedw-4
     do y=1,SIZE(OUT,1)
        if (is_NaN(OUT(y,i-4,j-4))) then
           OUT(y,i-4,j-4) = 0.0
        endif
     enddo
  enddo; enddo

  deallocate(tmp1)
  deallocate(tmp2)
  deallocate(tmp3)
  
end subroutine CNN_forward

subroutine ANN_forward(IN, OUT, weights1, weights2, weights3, weights4, G)
  real, dimension(:,:,:), intent(in) :: IN
  real, dimension(:,:,:), intent(inout) :: OUT
  real, dimension(:), intent(in) :: weights1
  real, dimension(:), intent(in) :: weights2
  real, dimension(:), intent(in) :: weights3
  real, dimension(:), intent(in) :: weights4
  type(SIS_hor_grid_type), intent(in) :: G
  real, dimension(32) :: tmp1
  real, dimension(64) :: tmp2
  real, dimension(128) :: tmp3

  integer :: i, j, x, y, z, is, ie, js, je
  
  is = G%isc ; ie = G%iec ; js = G%jsc ; je = G%jec
  
  OUT = 0.0
  do j=js,je ; do i=is,ie
     z = 1
     tmp1(:) = 0.0 ; tmp2(:) = 0.0 ; tmp3(:) = 0.0
     do x=1,SIZE(IN,1)
        do y=1,32
           tmp1(y) = tmp1(y) + IN(x,i,j)*weights1(z)
           z = z + 1
        enddo
     enddo
     z = 1
     do x=1,32
        do y=1,64
           tmp2(y) = tmp2(y) + max(0.0,tmp1(x))*weights2(z)
           z = z + 1
        enddo
     enddo
     z = 1
     do x=1,64
        do y=1,128
           tmp3(y) = tmp3(y) + max(0.0,tmp2(x))*weights3(z)
           z = z + 1
        enddo
     enddo
     z = 1
     do x=1,128
        do y=1,SIZE(OUT,1)
           OUT(y,i,j) = OUT(y,i,j) + max(0.0,tmp3(x))*weights4(z)
           z = z + 1
        enddo
     enddo
     do y=1,SIZE(OUT,1)
        if (is_NaN(OUT(y,i,j))) then
           OUT(y,i,j) = 0.0
        endif
     enddo
  enddo; enddo

end subroutine ANN_forward
  
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
  type(CNN_CS),              intent(inout)  :: CNN     !< Control structure for CNN
  real,                      intent(in)     :: dt_slow !< The thermodynamic time step [T ~> s]

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
                                   ::  WH_SSS    !< sea-surface salinity [psu].
  real, dimension(SZIW_(CNN),SZJW_(CNN)) &
                                   ::  WH_mask   !< land-sea mask (0=land cells, 1=ocean cells)
  real, dimension(8,SZIW_(CNN),SZJW_(CNN)) &
                                   ::  XA        !< input variables to network A (predict dsiconc)
  real, dimension(:,:,:), allocatable  &
                                   ::  XB        !< input variables to network B (predict dCN)

  !initialise network outputs
  real, dimension(:,:,:), allocatable &
                                   :: dSIC        !< network A predictions of aggregate SIC corrections
  real, dimension(:,:,:), allocatable &
                                   :: dCN         !< network B predictions of category SIC corrections
  real, dimension(SZI_(G),SZJ_(G),0:5) &
                                   :: posterior   !< updated part_size (bounded between 0 and 1)
  
  integer :: i, j, k, m, iT, jT
  integer :: is, ie, js, je, ncat, nlay
  integer :: isdw, iedw, jsdw, jedw
  real    :: cvr, Ti, qi_new, sic_inc
  real    :: rho_ice, Cp_water
  real    :: dists, positives
  integer :: dimX, dimY

  real :: hmid(5) = [0.05,0.2,0.5,0.9,2.0] !ITD thicknesses for new ice
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
  dimX = SIZE(XA,2) - 2*CNN%CNN_halo_size
  dimY = SIZE(XA,3) - 2*CNN%CNN_halo_size
  
  allocate(XB(7,dimX,dimY))
  allocate(dSIC(1,dimX,dimY))
  allocate(dCN(ncat,dimX,dimY))

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
  call pass_var(WH_SIC, CNN%CNN_Domain)
  call pass_var(WH_SST, CNN%CNN_Domain)
  call pass_vector(WH_UI, WH_VI, CNN%CNN_Domain, stagger=CGRID_NE)
  call pass_var(WH_HI, CNN%CNN_Domain)
  call pass_var(WH_TS, CNN%CNN_Domain)
  call pass_var(WH_SSS, CNN%CNN_Domain)
  call pass_var(WH_mask, CNN%CNN_Domain)
  
  XA = 0.0
  ! Combine arrays for CNN input
  do j=jsdw,jedw ; do i=isdw,iedw
     if (G%mask2dT(i,j) == 1.0) then !is ocean
        XA(1,i,j) = (WH_SIC(i,j) - sic_mu)/sic_std
        XA(2,i,j) = (WH_SST(i,j) - sst_mu)/sst_std
        XA(3,i,j) = (WH_UI(i,j) - ui_mu)/ui_std
        XA(4,i,j) = (WH_VI(i,j) - vi_mu)/vi_std
        XA(5,i,j) = (WH_HI(i,j) - hi_mu)/hi_std
        XA(6,i,j) = (WH_TS(i,j) - ts_mu)/ts_std
        XA(7,i,j) = (WH_SSS(i,j) - sss_mu)/sss_std
        XA(8,i,j) = WH_mask(i,j)
     endif
  enddo ; enddo

  dSIC = 0.0
  call CNN_forward(XA, dSIC, CNN%netA_weight_vec1, CNN%netA_weight_vec2, CNN%netA_weight_vec3, CNN%netA_weight_vec4, CNN)
  
  XB = 0.0
  do j=js,je ; do i=is,ie
     iT = i-CNN%CNN_halo_size
     jT = j-CNN%CNN_halo_size
     if (G%mask2dT(i,j) == 1.0) then !is ocean
        XB(1,iT,jT) = (dSIC(1,iT,jT) - dsic_mu)/dsic_std
        XB(2,iT,jT) = (IST%part_size(i,j,1) - cn1_mu)/cn1_std
        XB(3,iT,jT) = (IST%part_size(i,j,2) - cn2_mu)/cn2_std
        XB(4,iT,jT) = (IST%part_size(i,j,3) - cn3_mu)/cn3_std
        XB(5,iT,jT) = (IST%part_size(i,j,4) - cn4_mu)/cn4_std
        XB(6,iT,jT) = (IST%part_size(i,j,5) - cn5_mu)/cn5_std
        XB(7,iT,jT) = G%mask2dT(i,j)
     endif
  enddo; enddo

  dCN = 0.0
  call ANN_forward(XB, dCN, CNN%netB_weight_vec1, CNN%netB_weight_vec2, CNN%netB_weight_vec3, CNN%netB_weight_vec4, G)

  do j=js,je ; do i=is,ie
     iT = i-CNN%CNN_halo_size
     jT = j-CNN%CNN_halo_size
     do k=1,ncat
        if (G%mask2dT(i,j) == 1.0) then !is ocean
           IST%dCN(i,j,k) = dCN(k,iT,jT)!/(432000.0/dt_slow) !432000 = 5 days.
        endif
     enddo
  enddo; enddo

  deallocate(dCN)
  deallocate(dSIC)
  deallocate(XB)
  !call pass_var(IST%dCN, G%Domain)

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
