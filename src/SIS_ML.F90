!> This code performs a state-dependent bias correction to each sea ice concentration category (part_size).
!> The bias correction method was developed by training a Machine Learning (ML) model to predict sea ice
!> concentration data assimilation increments, using model state variables. An initial CNN architecture is
!> used to find a mapping from model state variables to the aggregate (observable) SIC increment. This prediction
!> is then passed to an ANN, along with other state variables, to predict a correction to each category concentration.
!> Details of this approach can be found at: https://doi.org/10.1029/2023MS003757.

!> The implementation of this bias-correction framework was originally shown in https://doi.org/10.1029/2023GL106776.
!> However this approach was based on updating model restart files offline (high I/O). This present code applies the corrections
!> at the thermodynamic timestep of SIS2. Due to the relatively simple CNN and ANN architectures, these have been
!> directly coded into Fortran here (see CNN_forward and ANN_forward subroutines below). More complicated architectures
!> will be easier to implement through Python wrappers such as Forpy or FTorch. Versions of this code have been developed
!> for both Forpy and FTorch and can be found at https://github.com/William-gregory/SIS2/tree/forpy_SPEAR and
!> https://github.com/William-gregory/SIS2/tree/ftorch_SPEAR, respectively.

!< Author: Will Gregory (wg4031@princeton.edu / william.gregory@noaa.gov)
!<
!<
!<

module SIS_ML

use ice_grid,                  only : ice_grid_type
use mpp_domains_mod,           only : domain2d
use SIS_hor_grid,              only : SIS_hor_grid_type
use MOM_io,                    only : MOM_read_data
use fms_io_mod,                only : register_restart_field, restart_file_type, restore_state
use MOM_domains,               only : clone_MOM_domain,MOM_domain_type
use MOM_domains,               only : pass_var, pass_vector, CGRID_NE
use SIS_diag_mediator,         only : SIS_diag_ctrl
use SIS_diag_mediator,         only : register_diag_field=>register_SIS_diag_field
use SIS_diag_mediator,         only : post_SIS_data, post_data=>post_SIS_data
use SIS2_ice_thm,              only : get_SIS2_thermo_coefs
use SIS_types,                 only : ice_state_type, ocean_sfc_state_type, fast_ice_avg_type, ice_ocean_flux_type
use MOM_file_parser,           only : get_param, param_file_type
use MOM_time_manager,          only : time_type


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

public :: ML_init,register_ML_restarts,ML_inference

!> Control structure for ML model
type, public :: ML_CS
  type(MOM_domain_type), pointer :: CNN_Domain => NULL()  !< Domain for inputs/outputs for the CNN
  integer :: isdw !< The lower i-memory limit for the wide halo arrays.
  integer :: iedw !< The upper i-memory limit for the wide halo arrays.
  integer :: jsdw !< The lower j-memory limit for the wide halo arrays.
  integer :: jedw !< The upper j-memory limit for the wide halo arrays.
  integer :: CNN_halo_size  !< Halo size at each side of subdomains
  real    :: ML_freq !< frequency of ML corrections

  !< The network weights for both CNN and ANN were raveled into a single vector offline.
  !< See https://github.com/William-gregory/FTorch/tree/SIS2/weights/Torch_to_netcdf.py
  !< TO DO: Generalize code to take any size weight vectors (or matrices?)
  real, dimension(2592)  :: CNN_weight_vec1 !< 9 x 32 x 3 x 3
  real, dimension(18432) :: CNN_weight_vec2 !< 32 x 64 x 3 x 3
  real, dimension(73728) :: CNN_weight_vec3 !< 64 x 128 x 3 x 3
  real, dimension(1152)  :: CNN_weight_vec4 !< 128 x 1 x 3 x 3
  real, dimension(224)   :: ANN_weight_vec1 !< 7 x 32
  real, dimension(2048)  :: ANN_weight_vec2 !< 32 x 64
  real, dimension(8192)  :: ANN_weight_vec3 !< 64 x 128
  real, dimension(640)   :: ANN_weight_vec4 !< 128 x 5

  character(len=120)  :: restart_file !< name of ice restart file(s)
  character(len=300)  :: CNN_weights !< filename of CNN weights netcdf file
  character(len=300)  :: ANN_weights !< filename of ANN weights netcdf file

  real :: count !< keeps track of 5-day time window for averaging
  real, dimension(:,:,:), allocatable :: &
       CN_filtered, &      !< Time-filtered category sea ice concentration [nondim]
       dCN_restart         !< Category sea ice concentration increments [nondim]
  real, dimension(:,:), allocatable :: &
       SIC_filtered, &  !< Time-filtered aggregate sea ice concentration [nondim]
       SST_filtered, &  !< Time-filtered sea-surface temperature [degC]
       UI_filtered, &   !< Time-filtered zonal ice velocities [ms-1]
       VI_filtered, &   !< Time-filtered meridional ice velocities [degC]
       HI_filtered, &   !< Time-filtered ice thickness [m]
       SW_filtered, &   !< Time-filtered net shortwave radiation [Wm-2]
       TS_filtered, &   !< Time-filtered ice-surface skin temperature [degC]
       SSS_filtered     !< Time-filtered sea-surface salinity [psu]
  
  type(SIS_diag_ctrl), pointer :: diag => NULL() !< A type that regulates diagnostics output
  !>@{ Diagnostic handles
  integer :: id_dcn = -1, id_sicnet = -1, id_sstnet = -1, id_uinet = -1, id_vinet = -1
  integer :: id_hinet = -1, id_swnet = -1, id_tsnet = -1, id_sssnet = -1, id_cnnet = -1
  !>@}
  
end type ML_CS

contains

!> Initialize ML routine and load CNN+ANN weights
subroutine ML_init(Time, G, param_file, diag, CS)
  type(time_type),               intent(in)    :: Time       !< Current model time  
  type(SIS_hor_grid_type),       intent(in)    :: G          !< The horizontal grid structure.
  type(param_file_type),         intent(in)    :: param_file !< Parameter file parser structure.
  type(SIS_diag_ctrl), target,   intent(inout) :: diag       !< Diagnostics structure.
  type(ML_CS),                   intent(inout) :: CS         !< Control structure for the ML model(s)

  ! Local Variables
  integer :: wd_halos(2) ! Varies with CNN
  real, parameter :: missing = -1e34
  character(len=40)  :: mdl = "SIS_ML"  ! module name

  ! Register fields for output from this module.
  CS%diag => diag
  CS%id_dcn    = register_diag_field('ice_model', 'dCN', diag%axesTc, Time, &
       'ML-based correction to ice concentration', 'area fraction', missing_value=missing)
  CS%id_sicnet    = register_diag_field('ice_model', 'SICnet', diag%axesT1, Time, &
       'Aggregate sea ice concentration CNN input', 'area fraction', missing_value=missing)
  CS%id_sstnet    = register_diag_field('ice_model', 'SSTnet', diag%axesT1, Time, &
       'Sea-surface temperature CNN input', 'deg C', missing_value=missing)
  CS%id_uinet    = register_diag_field('ice_model', 'UInet', diag%axesT1, Time, &
       'Zonal ice velocity CNN input', 'm s-1', missing_value=missing)
  CS%id_vinet    = register_diag_field('ice_model', 'VInet', diag%axesT1, Time, &
       'Meridional ice velocity CNN input', 'm s-1', missing_value=missing)
  CS%id_hinet    = register_diag_field('ice_model', 'HInet', diag%axesT1, Time, &
       'Sea ice thickness CNN input', 'm', missing_value=missing)
  CS%id_swnet    = register_diag_field('ice_model', 'SWnet', diag%axesT1, Time, &
       'Net shortwave radiation CNN input', 'W m-2', missing_value=missing)
  CS%id_tsnet    = register_diag_field('ice_model', 'TSnet', diag%axesT1, Time, &
       'Surface-skin temperature CNN input', 'deg C', missing_value=missing)
  CS%id_sssnet    = register_diag_field('ice_model', 'SSSnet', diag%axesT1, Time, &
       'Sea-surface salinity CNN input', 'g kg-1', missing_value=missing)
  CS%id_cnnet    = register_diag_field('ice_model', 'CNnet', diag%axesTc, Time, &
       'Category sea ice concentration CNN input', 'area fraction', missing_value=missing)
 
  call get_param(param_file, mdl, "RESTARTFILE", CS%restart_file, &
                 "The name of the restart file.", default="ice_model.res.nc")

  call get_param(param_file, mdl, "CNN_HALO_SIZE", CS%CNN_halo_size, &
      "Halo size at each side of subdomains, depends on CNN architecture.", & 
      units="nondim", default=4)

  call get_param(param_file, mdl, "ML_FREQ", CS%ML_freq, &
      "Frequency of applying ML corrections", units="seconds", default=86400.0)

  call get_param(param_file, mdl, "CNN_WEIGHTS", CS%CNN_weights, &
      "CNN optimized weights", &
      default="/gpfs/f5/scratch/gfdl_o/William.Gregory/FTorch/weights/NetworkA_weights.nc")

  call get_param(param_file, mdl, "ANN_WEIGHTS", CS%ANN_weights, &
      "ANN optimized weights", &
      default="/gpfs/f5/scratch/gfdl_o/William.Gregory/FTorch/weights/NetworkB_weights.nc")

  call MOM_read_data(filename=trim(CS%CNN_weights), fieldname="C1", data=CS%CNN_weight_vec1)
  call MOM_read_data(filename=trim(CS%CNN_weights), fieldname="C2", data=CS%CNN_weight_vec2)
  call MOM_read_data(filename=trim(CS%CNN_weights), fieldname="C3", data=CS%CNN_weight_vec3)
  call MOM_read_data(filename=trim(CS%CNN_weights), fieldname="C4", data=CS%CNN_weight_vec4)
  call MOM_read_data(filename=trim(CS%ANN_weights), fieldname="C1", data=CS%ANN_weight_vec1)
  call MOM_read_data(filename=trim(CS%ANN_weights), fieldname="C2", data=CS%ANN_weight_vec2)
  call MOM_read_data(filename=trim(CS%ANN_weights), fieldname="C3", data=CS%ANN_weight_vec3)
  call MOM_read_data(filename=trim(CS%ANN_weights), fieldname="C4", data=CS%ANN_weight_vec4)
  
  wd_halos(1) = CS%CNN_halo_size
  wd_halos(2) = CS%CNN_halo_size
  call clone_MOM_domain(G%Domain, CS%CNN_Domain, min_halo=wd_halos, symmetric=G%symmetric)
  CS%isdw = G%isc-wd_halos(1) ; CS%iedw = G%iec+wd_halos(1)
  CS%jsdw = G%jsc-wd_halos(2) ; CS%jedw = G%jec+wd_halos(2)

  if (.not. allocated(CS%SIC_filtered)) &
       allocate(CS%SIC_filtered(CS%isdw:CS%iedw,CS%jsdw:CS%jedw), source=0.)
  if (.not. allocated(CS%SST_filtered))	&
       allocate(CS%SST_filtered(CS%isdw:CS%iedw,CS%jsdw:CS%jedw), source=0.)
  if (.not. allocated(CS%UI_filtered))	&
       allocate(CS%UI_filtered(CS%isdw:CS%iedw,CS%jsdw:CS%jedw), source=0.)
  if (.not. allocated(CS%VI_filtered))	&
       allocate(CS%VI_filtered(CS%isdw:CS%iedw,CS%jsdw:CS%jedw), source=0.)
  if (.not. allocated(CS%SW_filtered))	&
       allocate(CS%HI_filtered(CS%isdw:CS%iedw,CS%jsdw:CS%jedw), source=0.)
  if (.not. allocated(CS%SW_filtered))	&
       allocate(CS%SW_filtered(CS%isdw:CS%iedw,CS%jsdw:CS%jedw), source=0.)
  if (.not. allocated(CS%TS_filtered))	&
       allocate(CS%TS_filtered(CS%isdw:CS%iedw,CS%jsdw:CS%jedw), source=0.)
  if (.not. allocated(CS%SSS_filtered))	&
       allocate(CS%SSS_filtered(CS%isdw:CS%iedw,CS%jsdw:CS%jedw), source=0.)
  if (.not. allocated(CS%CN_filtered))	&
       allocate(CS%CN_filtered(G%isc:G%iec,G%jsc:G%jec,5), source=0.)
  if (.not. allocated(CS%dCN_restart))	&
       allocate(CS%dCN_restart(G%isc:G%iec,G%jsc:G%jec,5), source=0.)
  CS%count = 1.

end subroutine ML_init

subroutine register_ML_restarts(CS, G, Ice_restart, restart_dir)
  type(ML_CS),             intent(in)    :: CS      !< Control structure for the ML model
  type(SIS_hor_grid_type), intent(in)    :: G       !< The horizontal grid type
  type(restart_file_type), pointer       :: Ice_restart !< A pointer to the restart type for the ice
  character(len=*),        intent(in)    :: restart_dir !< A directory in which to find the restart file

  integer :: id_cn, id_dcn, id_sic, id_sst, id_ui, id_vi, id_hi, id_sw, id_ts, id_sss, id_cnt
  type(domain2d), pointer :: mpp_domain => NULL()
  type(domain2d), pointer :: mpp_wh_domain => NULL()
  mpp_domain => G%Domain%mpp_domain
  mpp_wh_domain => CS%CNN_Domain%mpp_domain

  id_cn = register_restart_field(Ice_restart, trim(CS%restart_file), 'running_mean_cn',  CS%CN_filtered, domain=mpp_domain, mandatory=.false.)
  id_dcn = register_restart_field(Ice_restart, trim(CS%restart_file), 'part_size_increments',  CS%dCN_restart, domain=mpp_domain, mandatory=.false.)
  id_sic = register_restart_field(Ice_restart, trim(CS%restart_file), 'running_mean_sic', CS%SIC_filtered, domain=mpp_wh_domain, mandatory=.false.)
  id_sst = register_restart_field(Ice_restart, trim(CS%restart_file), 'running_mean_sst', CS%SST_filtered, domain=mpp_wh_domain, mandatory=.false.)
  id_ui = register_restart_field(Ice_restart, trim(CS%restart_file), 'running_mean_ui',  CS%UI_filtered, domain=mpp_wh_domain, mandatory=.false.)
  id_vi = register_restart_field(Ice_restart, trim(CS%restart_file), 'running_mean_vi',  CS%VI_filtered, domain=mpp_wh_domain, mandatory=.false.)
  id_hi = register_restart_field(Ice_restart, trim(CS%restart_file), 'running_mean_hi',  CS%HI_filtered, domain=mpp_wh_domain, mandatory=.false.)
  id_sw = register_restart_field(Ice_restart, trim(CS%restart_file), 'running_mean_sw',  CS%SW_filtered, domain=mpp_wh_domain, mandatory=.false.)
  id_ts = register_restart_field(Ice_restart, trim(CS%restart_file), 'running_mean_ts',  CS%TS_filtered, domain=mpp_wh_domain, mandatory=.false.)
  id_sss = register_restart_field(Ice_restart, trim(CS%restart_file), 'running_mean_sss', CS%SSS_filtered, domain=mpp_wh_domain, mandatory=.false.)
  id_cnt = register_restart_field(Ice_restart, trim(CS%restart_file), 'day_counter', CS%count, mandatory=.false.)

  call restore_state(Ice_restart, id_cn, restart_dir, nonfatal_missing_files=.true.)
  call restore_state(Ice_restart, id_dcn, restart_dir, nonfatal_missing_files=.true.)
  call restore_state(Ice_restart, id_sic, restart_dir, nonfatal_missing_files=.true.)
  call restore_state(Ice_restart, id_sst, restart_dir, nonfatal_missing_files=.true.)
  call restore_state(Ice_restart, id_ui, restart_dir, nonfatal_missing_files=.true.)
  call restore_state(Ice_restart, id_vi, restart_dir, nonfatal_missing_files=.true.)
  call restore_state(Ice_restart, id_hi, restart_dir, nonfatal_missing_files=.true.)
  call restore_state(Ice_restart, id_sw, restart_dir, nonfatal_missing_files=.true.)
  call restore_state(Ice_restart, id_ts, restart_dir, nonfatal_missing_files=.true.)
  call restore_state(Ice_restart, id_sss, restart_dir, nonfatal_missing_files=.true.)
  call restore_state(Ice_restart, id_cnt, restart_dir, nonfatal_missing_files=.true.)
  
end subroutine register_ML_restarts

!< The CNN loops over each grid point in the output domain (G%isc:G%iec, G%jsc:G%jec)
!< and performs 4 convolution operations to make a prediction at that grid point.
!< The current CNN architecture uses kernels of size 3x3. Therefore for 4 convolution
!< operations a 9x9 stencil is needed per grid point. While code has been generalized
!< to pad data with an arbitrary halo size (given by CS%CNN_halo_size), the CNN_forward
!< subroutine still assumes a halo size of 4. Therefore from an initial 9x9 domain, the
!< first convolution outputs a 7x7 domain, then the next outputs a 5x5 domain and so on,
!< until the prediction at grid point i,j. Note that the first 3 convolution operations
!< are then passed through a ReLU function, given by max(0.0,x).
subroutine CNN_forward(IN, OUT, weights1, weights2, weights3, weights4, G)
  real, dimension(:,:,:), intent(in)  ::  IN
  real, dimension(:,:), intent(inout) :: OUT
  real, dimension(:), intent(in) :: weights1
  real, dimension(:), intent(in) :: weights2
  real, dimension(:), intent(in) :: weights3
  real, dimension(:), intent(in) :: weights4
  type(SIS_hor_grid_type), intent(in) :: G
  
  real, dimension(32,7,7)  :: tmp1
  real, dimension(64,5,5)  :: tmp2
  real, dimension(128,3,3) :: tmp3
  integer :: i, j, x, y, z, u, v, m, n, is, ie, js, je

  is = G%isc ; ie = G%iec ; js = G%jsc ; je = G%jec
  
  do j=js,je ; do i=is,ie
     tmp1 = 0.0 ; tmp2 = 0.0 ; tmp3 = 0.0
     do m=-3,3 !loop over 7x7 stencil in x-direction
        do n=-3,3 !loop over 7x7 stencil in y-direction
           z = 1 !z is a index tracker for the weights
           do x=1,SIZE(IN,1) !loop over input features (SIC,SST,UI,... etc)
              do y=1,32 !loop over the number of features in the first layer
                 do u=-1,1 !loop over convolution kernel in x-direction 
                    do v=-1,1 !loop over convolution kernel in y-direction
                       tmp1(y,m+4,n+4) = tmp1(y,m+4,n+4) + (IN(x,i+m+u,j+n+v)*weights1(z))
                       z = z + 1
                    enddo
                 enddo
              enddo
           enddo
        enddo
     enddo
     do m=-2,2
        do n=-2,2
           z = 1
           do x=1,32
              do y=1,64
                 do u=-1,1
                    do v=-1,1
                       tmp2(y,m+3,n+3) = tmp2(y,m+3,n+3) + (max(0.0,tmp1(x,m+4+u,n+4+v))*weights2(z))
                       z = z + 1
                    enddo
                 enddo
              enddo
           enddo
        enddo
     enddo
     do m=-1,1
        do n=-1,1
           z = 1
           do x=1,64
              do y=1,128
                 do u=-1,1
                    do v=-1,1
                       tmp3(y,m+2,n+2) = tmp3(y,m+2,n+2) + (max(0.0,tmp2(x,m+3+u,n+3+v))*weights3(z))
                       z = z + 1
                    enddo
                 enddo
              enddo
           enddo
        enddo
     enddo
     z = 1
     do x=1,128
        do u=1,3
           do v=1,3
              OUT(i,j) = OUT(i,j) + (max(0.0,tmp3(x,u,v))*weights4(z))
              z = z + 1
           enddo
        enddo
     enddo
  enddo; enddo
  
end subroutine CNN_forward

!< The ANN routine is much more straightforward to implement as we don't need to
!< consider halos or kernels. We simply loop over the grid and perform local linear
!< weighted sums.
!< TO DO: make general to any number of network layers, and width of each layer?
subroutine ANN_forward(IN, OUT, weights1, weights2, weights3, weights4, G)
  real, dimension(:,:,:), intent(in)    ::  IN
  real, dimension(:,:,:), intent(inout) :: OUT
  real, dimension(:), intent(in) :: weights1
  real, dimension(:), intent(in) :: weights2
  real, dimension(:), intent(in) :: weights3
  real, dimension(:), intent(in) :: weights4
  type(SIS_hor_grid_type), intent(in) :: G
  real, dimension(32)  :: tmp1
  real, dimension(64)  :: tmp2
  real, dimension(128) :: tmp3

  integer :: i, j, x, y, z, is, ie, js, je
  
  is = G%isc ; ie = G%iec ; js = G%jsc ; je = G%jec
  
  do j=js,je ; do i=is,ie
     z = 1
     tmp1 = 0.0 ; tmp2 = 0.0 ; tmp3 = 0.0
     do x=1,SIZE(IN,1)
        do y=1,32
           tmp1(y) = tmp1(y) + (IN(x,i,j)*weights1(z))
           z = z + 1
        enddo
     enddo
     z = 1
     do x=1,32
        do y=1,64
           tmp2(y) = tmp2(y) + (max(0.0,tmp1(x))*weights2(z))
           z = z + 1
        enddo
     enddo
     z = 1
     do x=1,64
        do y=1,128
           tmp3(y) = tmp3(y) + (max(0.0,tmp2(x))*weights3(z))
           z = z + 1
        enddo
     enddo
     z = 1
     do x=1,128
        do y=1,SIZE(OUT,3)
           OUT(i,j,y) = OUT(i,j,y) + (max(0.0,tmp3(x))*weights4(z))
           z = z + 1
        enddo
     enddo
  enddo; enddo

end subroutine ANN_forward

subroutine postprocess(IST, increments, G, IG)
  type(ice_state_type),       intent(inout)  :: IST     !< A type describing the state of the sea ice
  real, dimension(:,:,:),     intent(in)     :: increments !< ML-predicted increments
  type(SIS_hor_grid_type),    intent(in)     :: G       !< The horizontal grid structure
  type(ice_grid_type),        intent(in)     :: IG      !< Sea ice specific grid

  real, dimension(SZI_(G),SZJ_(G),0:5) &
       :: posterior  !< updated part_size (bounded between 0 and 1)

  logical, dimension(5) :: negatives
  real    :: dists, positives
  integer :: i, j, k, m
  integer :: is, ie, js, je, ncat, nlay
  real    :: cvr, Tf, enth_new
  real    :: irho_ice, rho_ice
  real :: hmid(5) = [0.05,0.2,0.5,0.9,2.0] !ITD thicknesses for new ice

  !parameters for adding new sea ice to a grid cell which was previously ice free
  real, parameter :: & 
       phi_init = 0.75, & !initial liquid fraction of frazil ice
       Si_new = 5.0    !salinity of mushy ice (ppt)

  call get_SIS2_thermo_coefs(IST%ITV, rho_ice=rho_ice)
  is = G%isc ; ie = G%iec ; js = G%jsc ; je = G%jec ; ncat = IG%CatIce ; nlay = IG%NkIce
  irho_ice = 1/rho_ice

  !Update category concentrations & bound between 0 and 1
  !This part checks if the updated SIC in any category is below zero.
  !If it is, spread the equivalent negative value across the other positive categories
  !E.g if new SIC is [-0.2,0.1,0.2,0.3,0.4], then remove 0.2/4 from categories 2 through 5
  !E.g if new SIC is [-0.2,-0.1,0.4,0.2,0.1], then remove 0.3/3 from categories 3 through 5
  !This will continue in a 'while loop' until all categories are >= 0.
  posterior = 0.0
  do j=js,je ; do i=is,ie
     do k=1,ncat
        posterior(i,j,k) = IST%part_size(i,j,k) + increments(i,j,k)
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
              posterior(i,j,k) = posterior(i,j,k) - (dists/positives)
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
  !see https://github.com/CICE-Consortium/Icepack/blob/main/columnphysics/icepack_therm_itd.F90
  Tf = min(liquidus_temperature_mush(Si_new/phi_init),-0.1)
  enth_new = enthalpy_ice(Tf, Si_new)
  do j=js,je ; do i=is,ie
     do k=1,ncat
        !have added ice to grid cell which was previously ice free
        if (posterior(i,j,k)>0.0 .and. IST%part_size(i,j,k)<=0.0) then
           IST%mH_ice(i,j,k) = hmid(k)*rho_ice
           IST%mH_snow(i,j,k) = 0.0
           IST%mH_pond(i,j,k) = 0.0
           IST%enth_snow(i,j,k,1) = 0.0
           do m=1,nlay
              IST%enth_ice(i,j,k,m) = enth_new*irho_ice
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
     enddo
     IST%part_size(i,j,0) = posterior(i,j,0)
  enddo; enddo

end subroutine postprocess

  
!> This routine does all of the data prep for both the CNN and ANN, including padding the data
!> for the CNN, and normalizing all inputs. The predicted increments are the added to the prior
!> sea ice concentration states and a post-processing step then bounds this new (posterior) state
!> between 0 and 1, and then makes commensurate adjustments to the sea ice profiles in the case of
!> adding/removing sea ice (i.e add thickness and salinity for new ice). The code is currently non-
!> conservative in terms of heat, mass, salt.
subroutine ML_inference(IST, FIA, OSS, G, IG, ML, dt_slow)
  type(ice_state_type),       intent(inout)  :: IST     !< A type describing the state of the sea ice
  type(fast_ice_avg_type),    intent(in)     :: FIA     !< A type containing averages of fields
                                                        ! (mostly fluxes) over the fast updates
  type(ocean_sfc_state_type), intent(in)     :: OSS     !< A structure containing the arrays that describe
                                                        !  the ocean's surface state for the ice model.
  type(SIS_hor_grid_type),    intent(in)     :: G       !< The horizontal grid structure
  type(ice_grid_type),        intent(in)     :: IG      !< Sea ice specific grid
  type(ML_CS) ,               intent(inout)  :: ML      !< Control structure for the ML model
  real,                       intent(in)     :: dt_slow !< The thermodynamic time step [T ~> s]

  real, dimension(9,SZIW_(ML),SZJW_(ML)) &
                                   ::  IN_CNN    !< input variables to CNN (predict dSIC)
  real, dimension(7,SZI_(G),SZJ_(G)) &
                                   ::  IN_ANN    !< input variables to ANN (predict dCN)

  real, dimension(SZI_(G),SZJ_(G)) &
                                   :: dSIC       !< CNN predictions of aggregate SIC corrections
  real, dimension(SZI_(G),SZJ_(G),5) &
                                   :: dCN        !< ANN predictions of category SIC corrections
  real, dimension(SZIW_(ML),SZJW_(ML)) &
                                   :: land_mask  !< Land-sea mask [land cells = 0, ocean cells = 1]  
  
  integer :: i, j, k
  integer :: is, ie, js, je, ncat
  integer :: isdw, iedw, jsdw, jedw
  real    :: scale, nsteps
  
  !normalization statistics for both networks
  real, parameter :: &
       !CNN stats
       sic_mu = 0.29760098549490005, &
       sst_mu = 2.3628579351247665, &
       ui_mu = 0.05215740632978765, &
       vi_mu = 0.015774301594485004, &
       hi_mu = 0.3428559690813135, &
       sw_mu = 67.89703631265903, &
       ts_mu = -4.930865654514209, &
       sss_mu = 29.812795055984434, &

       sic_std = 2.3988684677904093, &
       sst_std = 0.19315381038814353, &
       ui_std = 8.089628019796052, & 
       vi_std = 11.506500421554342, & 
       hi_std = 1.68075870925751, &
       sw_std = 0.013607104867283745, &
       ts_std = 0.1180058324648759, & 
       sss_std = 0.09315672798399899, & 

       !ANN stats
       dsic_mu = -0.0011355808093858428, &
       cn1_mu = 0.013881755289701567, &
       cn2_mu = 0.04523203783883471, &
       cn3_mu = 0.09591018269427339, &
       cn4_mu = 0.05589209926886554, &
       cn5_mu = 0.12853458139886695, &
       
       dsic_std = 27.745509886748263, &
       cn1_std = 18.43686888204614, &
       cn2_std = 7.828396154351002, &
       cn3_std = 4.63604339451611, &
       cn4_std = 6.17786223701505, &
       cn5_std = 3.3852270028512286

  scale = ML%ML_freq/432000.0 !dt_slow/432000.0 !Network was trained on 5-day (432000-second) increments
  nsteps = ML%ML_freq/dt_slow !number of timesteps in ML%ML_freq
  is = G%isc ; ie = G%iec ; js = G%jsc ; je = G%jec ; ncat = IG%CatIce
  isdw = ML%isdw; iedw = ML%iedw; jsdw = ML%jsdw; jedw = ML%jedw

  dCN = 0.0 ; land_mask = 0.0
  do j=js,je ; do i=is,ie
     land_mask(i,j) = G%mask2dT(i,j)
     do k=1,ncat
        dCN(i,j,k) = ML%dCN_restart(i,j,k)
     enddo
  enddo; enddo

  !if ( (.not. all(dCN==0.0)) .and. (ML%count /= nsteps) ) then
  !   call postprocess(IST, dCN, G, IG)
  !endif
  
  if ( ML%count == nsteps ) then !nsteps have passed, do inference

     ! Update the wide halos
     call pass_var(ML%SIC_filtered, ML%CNN_Domain)
     call pass_var(ML%SST_filtered, ML%CNN_Domain)
     call pass_vector(ML%UI_filtered, ML%VI_filtered, ML%CNN_Domain, stagger=CGRID_NE)
     call pass_var(ML%HI_filtered, ML%CNN_Domain)
     call pass_var(ML%SW_filtered, ML%CNN_Domain)
     call pass_var(ML%TS_filtered, ML%CNN_Domain)
     call pass_var(ML%SSS_filtered, ML%CNN_Domain)        
     call pass_var(land_mask, ML%CNN_Domain)

     IN_CNN = 0.0
     ! Combine arrays for the CNN and normalize
     do j=jsdw,jedw ; do i=isdw,iedw
        IN_CNN(1,i,j) = land_mask(i,j) * ((ML%SIC_filtered(i,j) - sic_mu)*sic_std)
        IN_CNN(2,i,j) = land_mask(i,j) * ((ML%SST_filtered(i,j) - sst_mu)*sst_std)
        IN_CNN(3,i,j) = land_mask(i,j) * ((ML%UI_filtered(i,j) - ui_mu)*ui_std)
        IN_CNN(4,i,j) = land_mask(i,j) * ((ML%VI_filtered(i,j) - vi_mu)*vi_std)
        IN_CNN(5,i,j) = land_mask(i,j) * ((ML%HI_filtered(i,j) - hi_mu)*hi_std)
        IN_CNN(6,i,j) = land_mask(i,j) * ((ML%SW_filtered(i,j) - sw_mu)*sw_std)
        IN_CNN(7,i,j) = land_mask(i,j) * ((ML%TS_filtered(i,j) - ts_mu)*ts_std)
        IN_CNN(8,i,j) = land_mask(i,j) * ((ML%SSS_filtered(i,j) - sss_mu)*sss_std)
        IN_CNN(9,i,j) = land_mask(i,j)
     enddo ; enddo

     dSIC = 0.0
     call CNN_forward(IN_CNN, dSIC, ML%CNN_weight_vec1, ML%CNN_weight_vec2, ML%CNN_weight_vec3, ML%CNN_weight_vec4, G)

     IN_ANN = 0.0
     do j=js,je ; do i=is,ie
        IN_ANN(1,i,j) = G%mask2dT(i,j) * ((dSIC(i,j) - dsic_mu)*dsic_std)
        IN_ANN(2,i,j) = G%mask2dT(i,j) * ((ML%CN_filtered(i,j,1) - cn1_mu)*cn1_std)
        IN_ANN(3,i,j) = G%mask2dT(i,j) * ((ML%CN_filtered(i,j,2) - cn2_mu)*cn2_std)
        IN_ANN(4,i,j) = G%mask2dT(i,j) * ((ML%CN_filtered(i,j,3) - cn3_mu)*cn3_std)
        IN_ANN(5,i,j) = G%mask2dT(i,j) * ((ML%CN_filtered(i,j,4) - cn4_mu)*cn4_std)
        IN_ANN(6,i,j) = G%mask2dT(i,j) * ((ML%CN_filtered(i,j,5) - cn5_mu)*cn5_std)
        IN_ANN(7,i,j) = G%mask2dT(i,j)
     enddo; enddo

     dCN = 0.0
     call ANN_forward(IN_ANN, dCN, ML%ANN_weight_vec1, ML%ANN_weight_vec2, ML%ANN_weight_vec3, ML%ANN_weight_vec4, G)

     ML%dCN_restart(:,:,:) = 0.0
     do j=js,je ; do i=is,ie
        do k=1,ncat
           dCN(i,j,k) = G%mask2dT(i,j) * (dCN(i,j,k)*scale)
           ML%dCN_restart(i,j,k) = dCN(i,j,k)
        enddo
     enddo; enddo

     call postprocess(IST, dCN, G, IG)
     
     ML%SIC_filtered(:,:) = 0.0
     ML%SST_filtered(:,:) = 0.0
     ML%UI_filtered(:,:) = 0.0
     ML%VI_filtered(:,:) = 0.0
     ML%HI_filtered(:,:) = 0.0
     ML%SW_filtered(:,:) = 0.0
     ML%TS_filtered(:,:) = 0.0
     ML%SSS_filtered(:,:) = 0.0
     ML%CN_filtered(:,:,:) = 0.0
     ML%count = 0.
  endif

  if (ML%id_dcn>0) call post_data(ML%id_dcn, dCN, ML%diag)
  
  ML%count = ML%count + 1.

end subroutine ML_inference

! the functions below are taken from https://github.com/CICE-Consortium/Icepack/blob/main/columnphysics/icepack_mushy_physics.F90
! see also pages 57--62 of the CICE manual (https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=5eca93a8fbc716474f8fd80c804319b630f90316)
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
