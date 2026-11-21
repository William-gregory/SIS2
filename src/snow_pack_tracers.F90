! This file is part of the SIS2 sea ice model.
! See the LICENSE file for licensing information.
! SPDX-License-Identifier: Apache-2.0

!> State-dependent snow density and conductivity from tracked snow-pack properties.
!!
!! Three mass-weighted snow tracers (one layer per thickness category) are carried with the snow:
!!   snow_age_arr    Arrhenius-weighted age since deposition [days], referenced to -10 degC
!!   snow_wind_dose  accumulated wind-packing exposure, sum of max(U10 - 5, 0)^3 dt [m3 s-3 days]
!!   snow_wind_dep   10 m wind speed during deposition [m s-1]
!! plus a 2-d field that belongs to the atmosphere, not the snow, so it is not advected:
!!   vw_ema7         7-day exponential running mean of the 10 m wind speed [m s-1]
!!
!! Density is from equation discovery (PySR), fitted to SNOWPACK run on eORCA1 sea-ice columns:
!!   z   = b0 - exp(c_thin)/(1 + SWE) + w_age ln(1 + age_arr) + w_wind [ln(1 + wind_dose) + wind_dep + vw_ema7]
!!   rho = exp(log_rho_max) / (1 + exp(-z)),  clipped to [RHO_MIN, RHO_MAX]
!! with SWE the snow mass per unit ice area [kg m-2].
!!
!! Conductivity (SNOW_PACK_KS_SCHEME = SNOWPACK, the default), fitted to SNOWPACK's effective
!! conductivity of the layers in series, h_s / sum(dz_i / k_i):
!!   k = exp[ (rho/300)^p - c + g T ],  T = min(T_snow, 0) [degC],  p = 1.248, c = 2.462, g = 0.0151
!! with T_snow the daily-mean snow-layer temperature from enth_snow.
!!
!! Daily update.  Every slow step adds to daily accumulators (10 m wind, surface temperature, snow
!! temperature and snowfall).  Once a full day has accumulated, the tracers, the running-mean wind and
!! (optionally) the stochastic terms are advanced with the daily means, as in the training data, and
!! density and conductivity are recomputed.  They then stay fixed until the next daily update.
!!
!! Stochastic option (SNOW_PACK_STOCHASTIC).  z -> z + e, e = F^-1(Phi(eta)), with
!!   eta = sqrt(a) eta_p + sqrt(1 - a) eta_f,  clamped to +/- ETA_MAX
!! eta_f: AR(1) with time scale tau_f.  eta_p: changes only on snowfall, mixed with a fresh draw with
!! weight w = min(0.999, beta * new snow / snow mass), keeping unit variance.  Both are snow tracers;
!! a new pack starts with fresh draws.  F^-1(Phi(.)) is a table of the equation-B error quantiles.
!! Random numbers are counter-based (cell, category, day), so restarts and layouts reproduce exactly.
!!
!! SIS2 tracers are concentrations per unit snow mass and are advected weighted by snow mass, so the
!! values here are mass-weighted pack averages.  Melt, sublimation and snow-ice conversion remove
!! mass without changing the tracers; snowfall dilutes them with fresh snow.
module snow_pack_tracers

use MOM_error_handler,  only : SIS_error=>MOM_error, FATAL, WARNING
use MOM_file_parser,    only : get_param, log_version, param_file_type
use MOM_random,         only : random_01_CB
use MOM_time_manager,   only : time_type, time_type_to_real
use MOM_unit_scaling,   only : unit_scale_type
use SIS_diag_mediator,  only : register_SIS_diag_field, SIS_diag_ctrl, post_data=>post_SIS_data
use SIS_restart,        only : register_restart_field, SIS_restart_CS
use SIS_hor_grid,       only : SIS_hor_grid_type
use SIS_tracer_registry, only : register_SIS_tracer, SIS_tracer_registry_type
use SIS_types,          only : ice_state_type, fast_ice_avg_type
use SIS2_ice_thm,       only : Temp_from_En_S
use ice_grid,           only : ice_grid_type

implicit none ; private

#include <SIS2_memory.h>

public register_snow_pack_tracers, initialize_snow_pack_tracers
public snow_pack_step, snow_pack_post_diags, snow_pack_tracers_end

integer, parameter :: NTR = 3                     !< Number of deterministic snow tracers
integer, parameter :: I_AGE = 1, I_DOSE = 2, I_DEP = 3
integer, parameter :: I_ETA_P = 1, I_ETA_F = 2    !< Indices of the stochastic tracers
character(len=16), parameter :: tr_name(NTR) = [character(len=16) :: &
    "snow_age_arr", "snow_wind_dose", "snow_wind_dep"]
character(len=16), parameter :: tr_units(NTR) = [character(len=16) :: &
    "days", "m3 s-3 days", "m s-1"]
character(len=64), parameter :: tr_long(NTR) = [character(len=64) :: &
    "Arrhenius-weighted snow age", "Snow wind-packing exposure", "Wind speed at snow deposition"]
character(len=16), parameter :: eta_name(2) = [character(len=16) :: "snow_eta_p", "snow_eta_f"]

!> Quantiles of the equation-B error in z, e = F^-1(Phi(eta)) (SNOWPACK 1995-2023)
integer, parameter :: NQ = 11
real, parameter :: ETA_TAB(NQ) = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
real, parameter :: E_TAB(NQ)   = [-0.96, -0.70, -0.46, -0.27, -0.12, 0.03, 0.20, 0.40, 0.67, 1.11, 2.26]

!> Control structure for the snow-pack tracer package
type, public :: snow_pack_CS
  real, pointer :: tr(:,:,:,:,:) => NULL()     !< Snow tracers (i, j, cat, 1, NTR), units as above
  real, pointer :: eta(:,:,:,:,:) => NULL()    !< Stochastic snow tracers (i, j, cat, 1, 2) [nondim]
  real, allocatable :: vw_ema7(:,:)            !< 7-day running mean 10 m wind [m s-1]
  real, allocatable :: rho_cat(:,:,:)          !< Density from the last daily update [kg m-3]
  real, allocatable :: Ks_cat(:,:,:)           !< Conductivity from the last daily update [W m-1 K-1]

  ! Daily accumulators (restarted, so a day can span a restart)
  real :: t_accum = 0.0                        !< Time accumulated in the current day [s]
  real, allocatable :: U_sum(:,:)              !< Sum of U10 dt [m]
  real, allocatable :: Ts_sum(:,:,:)           !< Sum of surface temperature dt [K s]
  real, allocatable :: Tsn_sum(:,:,:)          !< Sum of snow temperature dt while snow exists [degC s]
  real, allocatable :: Tsn_t(:,:,:)            !< Time with snow in the current day [s]
  real, allocatable :: snow_sum(:,:,:)         !< Snowfall in the current day [kg m-2]
  real, allocatable :: m_start(:,:,:)          !< Snow mass at the start of the day [kg m-2]

  ! Equation B constants (nondimensional; rho_max in kg m-3 after exp)
  real :: log_rho_max, b0, w_age, w_wind, c_thin
  real :: rho_min, rho_max                     !< Safety clip [kg m-3]
  real :: Q_over_R                             !< Activation energy over gas constant [K]
  real :: T_ref                                !< Reference temperature for the age weighting [K]
  real :: U_pack                               !< Wind threshold for wind packing [m s-1]
  real :: tau_ema                              !< Time scale of the wind running mean [days]
  real :: rho_air, Cd_10m                      !< Converts stress to U10: U10 = sqrt(|tau|/(rho_air Cd))
  real :: day_length                           !< Length of the update interval [s]
  character(len=16) :: Ks_scheme               !< "SNOWPACK", "STURM97" or "CONSTANT"
  real :: Ks_p, Ks_c, Ks_gT                    !< SNOWPACK conductivity fit: exponent, offset [nondim]
                                               !! and temperature coefficient [degC-1]
  real :: Ks_const                             !< Conductivity for Ks_scheme = CONSTANT [W m-1 K-1]
  real :: IC_val(NTR)                          !< Initial tracer values when not restarting

  ! Stochastic option
  logical :: stochastic                        !< If true, perturb z with the stochastic error
  real :: st_a, st_tau_f, st_beta, st_eta_max  !< Persistent share, fast time scale [days], snowfall
                                               !! mixing strength [nondim], clamp on eta [nondim]
  integer :: st_seed                           !< Seed for the random number streams

  type(time_type), pointer :: Time => NULL()
  type(SIS_diag_ctrl), pointer :: diag => NULL()
  integer :: id_tr(NTR) = -1, id_ema = -1, id_rho = -1, id_Ks = -1, id_eta(2) = -1
end type snow_pack_CS

contains

!> Register the snow tracers for advection and restarts.  Returns false if the package is not used.
logical function register_snow_pack_tracers(G, IG, param_file, CS, diag, TrReg, Ice_restart)
  type(SIS_hor_grid_type),        intent(in) :: G     !< The horizontal grid type
  type(ice_grid_type),            intent(in) :: IG    !< The sea-ice specific grid type
  type(param_file_type),          intent(in) :: param_file !< Run-time parameters
  type(snow_pack_CS),             pointer    :: CS    !< Set to this package's control structure
  type(SIS_diag_ctrl),            target     :: diag  !< Diagnostic regulation
  type(SIS_tracer_registry_type), pointer    :: TrReg !< The SIS tracer registry
  type(SIS_restart_CS),           pointer    :: Ice_restart !< Ice restart control structure

# include "version_variable.h"
  character(len=40) :: mdl = "snow_pack_tracers"
  integer :: m

  register_snow_pack_tracers = .false.
  if (associated(CS)) then
    call SIS_error(WARNING, "register_snow_pack_tracers called with an associated control structure.")
    return
  endif
  allocate(CS)

  call log_version(param_file, mdl, version, &
      "State-dependent snow density and conductivity from tracked snow-pack properties, "//&
      "updated once a day from daily-mean forcing.")
  call get_param(param_file, mdl, "SNOW_PACK_LOG_RHO_MAX", CS%log_rho_max, &
      "Log of the maximum snow density in equation B.", units="nondim", default=6.4770)
  call get_param(param_file, mdl, "SNOW_PACK_B0", CS%b0, &
      "Offset of the logistic argument in equation B.", units="nondim", default=-2.8350)
  call get_param(param_file, mdl, "SNOW_PACK_W_AGE", CS%w_age, &
      "Weight on ln(1 + age_arr) in equation B.", units="nondim", default=0.2869)
  call get_param(param_file, mdl, "SNOW_PACK_W_WIND", CS%w_wind, &
      "Weight on the three wind terms in equation B.", units="nondim", default=0.0867)
  call get_param(param_file, mdl, "SNOW_PACK_C_THIN", CS%c_thin, &
      "Log of the thin-snow correction coefficient in equation B.", units="nondim", default=0.6254)
  call get_param(param_file, mdl, "SNOW_PACK_RHO_MIN", CS%rho_min, &
      "Lower safety limit on the snow density.", units="kg m-3", default=50.0)
  call get_param(param_file, mdl, "SNOW_PACK_RHO_MAX", CS%rho_max, &
      "Upper safety limit on the snow density.", units="kg m-3", default=650.0)
  call get_param(param_file, mdl, "SNOW_PACK_Q_OVER_R", CS%Q_over_R, &
      "Activation energy over the gas constant for the Arrhenius age weighting "//&
      "(67 kJ mol-1 / 8.314 J mol-1 K-1).", units="K", default=67000.0/8.314)
  call get_param(param_file, mdl, "SNOW_PACK_T_REF", CS%T_ref, &
      "Temperature at which one day of age counts as one day.", units="K", default=263.15)
  call get_param(param_file, mdl, "SNOW_PACK_U_PACK", CS%U_pack, &
      "Wind speed above which wind packing accumulates.", units="m s-1", default=5.0)
  call get_param(param_file, mdl, "SNOW_PACK_TAU_EMA", CS%tau_ema, &
      "Time scale of the running-mean wind speed.", units="days", default=7.0)
  call get_param(param_file, mdl, "SNOW_PACK_RHO_AIR", CS%rho_air, &
      "Air density used to convert wind stress to a 10 m wind speed.", units="kg m-3", default=1.275)
  call get_param(param_file, mdl, "SNOW_PACK_CD_10M", CS%Cd_10m, &
      "Neutral 10 m drag coefficient over ice used to convert wind stress to a 10 m wind speed. "//&
      "The default is (0.4 / ln(10 m / MOMENTUM_ROUGH_ICE))^2 for MOMENTUM_ROUGH_ICE = 1e-4 m.", &
      units="nondim", default=1.2e-3)
  call get_param(param_file, mdl, "SNOW_PACK_UPDATE_INTERVAL", CS%day_length, &
      "Interval between updates of the snow-pack properties, density and conductivity, "//&
      "which use means of the forcing over the interval.", units="s", default=86400.0)
  call get_param(param_file, mdl, "SNOW_PACK_KS_SCHEME", CS%Ks_scheme, &
      "Snow conductivity: SNOWPACK (fit to SNOWPACK, function of density and snow "//&
      "temperature), STURM97 (function of density) or CONSTANT.", default="SNOWPACK")
  call get_param(param_file, mdl, "SNOW_PACK_KS_P", CS%Ks_p, &
      "Density exponent in the SNOWPACK conductivity fit.", units="nondim", default=1.248)
  call get_param(param_file, mdl, "SNOW_PACK_KS_C", CS%Ks_c, &
      "Offset in the SNOWPACK conductivity fit.", units="nondim", default=2.462)
  call get_param(param_file, mdl, "SNOW_PACK_KS_GT", CS%Ks_gT, &
      "Snow-temperature coefficient in the SNOWPACK conductivity fit.", units="degC-1", default=0.0151)
  call get_param(param_file, mdl, "SNOW_PACK_KS_CONST", CS%Ks_const, &
      "Snow conductivity when SNOW_PACK_KS_SCHEME = CONSTANT.", units="W m-1 K-1", default=0.31)
  call get_param(param_file, mdl, "SNOW_PACK_AGE_IC", CS%IC_val(I_AGE), &
      "Initial snow age when not restarting.", units="days", default=0.0)
  call get_param(param_file, mdl, "SNOW_PACK_DOSE_IC", CS%IC_val(I_DOSE), &
      "Initial wind-packing exposure when not restarting.", units="m3 s-3 days", default=0.0)
  call get_param(param_file, mdl, "SNOW_PACK_WIND_DEP_IC", CS%IC_val(I_DEP), &
      "Initial deposition wind speed when not restarting.", units="m s-1", default=0.0)

  call get_param(param_file, mdl, "SNOW_PACK_STOCHASTIC", CS%stochastic, &
      "If true, add a stochastic error to the density equation, with a persistent part reset "//&
      "by snowfall and a fast AR(1) part. If false, the deterministic equation is used.", &
      default=.false.)
  call get_param(param_file, mdl, "SNOW_PACK_STOCH_A", CS%st_a, &
      "Share of the error variance in the persistent, snowfall-reset part.", &
      units="nondim", default=0.70, do_not_log=.not.CS%stochastic)
  call get_param(param_file, mdl, "SNOW_PACK_STOCH_TAU_F", CS%st_tau_f, &
      "Time scale of the fast AR(1) part of the error.", units="days", default=1.9, &
      do_not_log=.not.CS%stochastic)
  call get_param(param_file, mdl, "SNOW_PACK_STOCH_BETA", CS%st_beta, &
      "Snowfall mixing strength: the persistent part is mixed with a fresh draw with weight "//&
      "min(0.999, beta * daily new snow / snow mass).", units="nondim", default=8.5, &
      do_not_log=.not.CS%stochastic)
  call get_param(param_file, mdl, "SNOW_PACK_STOCH_ETA_MAX", CS%st_eta_max, &
      "Clamp on the standardised error before the quantile lookup.", units="nondim", default=2.5, &
      do_not_log=.not.CS%stochastic)
  call get_param(param_file, mdl, "SNOW_PACK_STOCH_SEED", CS%st_seed, &
      "Seed for the stochastic snow-density error, 0 to 999; change it for a different, "//&
      "independent ensemble member. The same seed reproduces a run exactly.", &
      default=0, do_not_log=.not.CS%stochastic)
  if (CS%stochastic .and. ((CS%st_seed < 0) .or. (CS%st_seed > 999))) &
    call SIS_error(FATAL, "snow_pack_tracers: SNOW_PACK_STOCH_SEED must be between 0 and 999.")

  if ((trim(CS%Ks_scheme) /= "SNOWPACK") .and. (trim(CS%Ks_scheme) /= "STURM97") .and. &
      (trim(CS%Ks_scheme) /= "CONSTANT")) &
    call SIS_error(FATAL, "snow_pack_tracers: unknown SNOW_PACK_KS_SCHEME "//trim(CS%Ks_scheme))

  allocate(CS%tr(SZI_(G), SZJ_(G), IG%CatIce, 1, NTR), source=0.0)
  allocate(CS%vw_ema7(SZI_(G), SZJ_(G)), source=0.0)
  allocate(CS%U_sum(SZI_(G), SZJ_(G)), source=0.0)
  allocate(CS%rho_cat(SZI_(G), SZJ_(G), IG%CatIce), source=0.0)
  allocate(CS%Ks_cat(SZI_(G), SZJ_(G), IG%CatIce), source=0.0)
  allocate(CS%Ts_sum(SZI_(G), SZJ_(G), IG%CatIce), source=0.0)
  allocate(CS%Tsn_sum(SZI_(G), SZJ_(G), IG%CatIce), source=0.0)
  allocate(CS%Tsn_t(SZI_(G), SZJ_(G), IG%CatIce), source=0.0)
  allocate(CS%snow_sum(SZI_(G), SZJ_(G), IG%CatIce), source=0.0)
  allocate(CS%m_start(SZI_(G), SZJ_(G), IG%CatIce), source=0.0)
  CS%diag => diag

  do m=1,NTR
    call register_restart_field(Ice_restart, trim(tr_name(m)), CS%tr(:,:,:,1,m), mandatory=.false.)
    ! A snow-only tracer: the matching Tr_ice slot keeps nL = 0, so ice-side loops skip it.
    call register_SIS_tracer(CS%tr(:,:,:,:,m), G, IG, 1, trim(tr_name(m)), param_file, TrReg, &
        snow_tracer=.true., massless_val=0.0, nonnegative=.true.)
  enddo
  if (CS%stochastic) then
    allocate(CS%eta(SZI_(G), SZJ_(G), IG%CatIce, 1, 2), source=0.0)
    do m=1,2
      call register_restart_field(Ice_restart, trim(eta_name(m)), CS%eta(:,:,:,1,m), mandatory=.false.)
      call register_SIS_tracer(CS%eta(:,:,:,:,m), G, IG, 1, trim(eta_name(m)), param_file, TrReg, &
          snow_tracer=.true., massless_val=0.0)
    enddo
  endif
  call register_restart_field(Ice_restart, "snow_vw_ema7", CS%vw_ema7, mandatory=.false.)
  call register_restart_field(Ice_restart, "snow_pack_rho", CS%rho_cat, mandatory=.false.)
  call register_restart_field(Ice_restart, "snow_pack_Ks", CS%Ks_cat, mandatory=.false.)
  call register_restart_field(Ice_restart, "snow_pack_t_accum", CS%t_accum, mandatory=.false.)
  call register_restart_field(Ice_restart, "snow_pack_U_sum", CS%U_sum, mandatory=.false.)
  call register_restart_field(Ice_restart, "snow_pack_Ts_sum", CS%Ts_sum, mandatory=.false.)
  call register_restart_field(Ice_restart, "snow_pack_Tsn_sum", CS%Tsn_sum, mandatory=.false.)
  call register_restart_field(Ice_restart, "snow_pack_Tsn_t", CS%Tsn_t, mandatory=.false.)
  call register_restart_field(Ice_restart, "snow_pack_snow_sum", CS%snow_sum, mandatory=.false.)
  call register_restart_field(Ice_restart, "snow_pack_m_start", CS%m_start, mandatory=.false.)

  register_snow_pack_tracers = .true.
end function register_snow_pack_tracers


!> Set initial values (when not restarting) and register diagnostics.
subroutine initialize_snow_pack_tracers(day, G, IG, CS, is_restart)
  type(time_type), target, intent(in) :: day        !< The current model time
  type(SIS_hor_grid_type), intent(in) :: G          !< The horizontal grid type
  type(ice_grid_type),     intent(in) :: IG         !< The sea-ice specific grid type
  type(snow_pack_CS),      pointer    :: CS         !< This package's control structure
  logical,                 intent(in) :: is_restart !< True if starting from a restart file
  integer :: m

  if (.not.associated(CS)) return
  CS%Time => day

  if (.not.is_restart) then
    do m=1,NTR ; CS%tr(:,:,:,1,m) = CS%IC_val(m) ; enddo
    CS%t_accum = 0.0     ! The accumulators and vw_ema7 start at zero; vw_ema7 is set on the first day.
  endif

  do m=1,NTR
    CS%id_tr(m) = register_SIS_diag_field("ice_model", trim(tr_name(m)), CS%diag%axesTc, &
        CS%Time, trim(tr_long(m)), trim(tr_units(m)), missing_value=-1.0e34)
  enddo
  CS%id_ema = register_SIS_diag_field("ice_model", "snow_vw_ema7", CS%diag%axesT1, CS%Time, &
      "7-day running-mean 10 m wind speed", "m s-1", missing_value=-1.0e34)
  CS%id_rho = register_SIS_diag_field("ice_model", "snow_rho_cat", CS%diag%axesTc, CS%Time, &
      "Snow density by category from the last daily update", "kg m-3", missing_value=-1.0e34)
  CS%id_Ks = register_SIS_diag_field("ice_model", "snow_Ks_cat", CS%diag%axesTc, CS%Time, &
      "Snow conductivity by category from the last daily update", "W m-1 K-1", missing_value=-1.0e34)
  if (CS%stochastic) then
    CS%id_eta(1) = register_SIS_diag_field("ice_model", "snow_eta_p", CS%diag%axesTc, CS%Time, &
        "Persistent part of the stochastic snow-density error", "nondim", missing_value=-1.0e34)
    CS%id_eta(2) = register_SIS_diag_field("ice_model", "snow_eta_f", CS%diag%axesTc, CS%Time, &
        "Fast part of the stochastic snow-density error", "nondim", missing_value=-1.0e34)
  endif
end subroutine initialize_snow_pack_tracers


!> Accumulate the forcing over one slow thermodynamic step; once a full day has accumulated,
!! advance the snow-pack properties and set the snow density and conductivity for the next day.
!! Call once per slow step, after the column thermodynamics.
subroutine snow_pack_step(CS, G, IG, US, IST, FIA, dt)
  type(snow_pack_CS),      pointer       :: CS  !< This package's control structure
  type(SIS_hor_grid_type), intent(in)    :: G   !< The horizontal grid type
  type(ice_grid_type),     intent(in)    :: IG  !< The sea-ice specific grid type
  type(unit_scale_type),   intent(in)    :: US  !< Unit conversion factors
  type(ice_state_type),    intent(inout) :: IST !< The ice state; rho_snow and Ks_snow are set here
  type(fast_ice_avg_type), intent(in)    :: FIA !< Averages of the fast fluxes, including wind stress
  real,                    intent(in)    :: dt  !< The step length [T ~> s]
  real :: dt_s, tau, Tsk
  integer :: i, j, k

  if (.not.associated(CS)) return
  dt_s = US%T_to_s * dt

  !$OMP parallel do default(shared) private(tau, Tsk)
  do j=G%jsc,G%jec ; do i=G%isc,G%iec
    tau = US%RLZ_T2_to_Pa * sqrt(FIA%WindStr_x(i,j)**2 + FIA%WindStr_y(i,j)**2)
    CS%U_sum(i,j) = CS%U_sum(i,j) + sqrt(tau / (CS%rho_air * CS%Cd_10m)) * dt_s
    do k=1,IG%CatIce
      ! Skin temperature averaged over the fast steps [C ~> degC]: by category if available
      ! (interpolated fluxes), otherwise the mean over the ice categories.  IST%t_surf is not used
      ! because it is only allocated on the slow side for EULERIAN_TSURF = False.
      if (allocated(FIA%Tskin_cat)) then
        Tsk = FIA%Tskin_cat(i,j,k)
      else
        Tsk = FIA%Tskin_avg(i,j)
      endif
      CS%Ts_sum(i,j,k) = CS%Ts_sum(i,j,k) + (US%C_to_degC*Tsk + 273.15) * dt_s
      CS%snow_sum(i,j,k) = CS%snow_sum(i,j,k) + US%RZ_to_kg_m2*max(FIA%fprec_top(i,j,k), 0.0) * dt
      if (IST%mH_snow(i,j,k) > 0.0) then
        CS%Tsn_sum(i,j,k) = CS%Tsn_sum(i,j,k) + &
            US%C_to_degC*Temp_from_En_S(IST%enth_snow(i,j,k,1), 0.0, IST%ITV) * dt_s
        CS%Tsn_t(i,j,k) = CS%Tsn_t(i,j,k) + dt_s
      endif
    enddo
  enddo ; enddo
  CS%t_accum = CS%t_accum + dt_s

  if (CS%t_accum >= CS%day_length - 0.5*dt_s) call snow_pack_daily_update(CS, G, IG, US, IST)
end subroutine snow_pack_step


!> Advance the snow-pack properties with the daily means and set density and conductivity.
subroutine snow_pack_daily_update(CS, G, IG, US, IST)
  type(snow_pack_CS),      pointer       :: CS  !< This package's control structure
  type(SIS_hor_grid_type), intent(in)    :: G   !< The horizontal grid type
  type(ice_grid_type),     intent(in)    :: IG  !< The sea-ice specific grid type
  type(unit_scale_type),   intent(in)    :: US  !< Unit conversion factors
  type(ice_state_type),    intent(inout) :: IST !< The ice state; rho_snow and Ks_snow are set here
  real :: dt_days, I_t, U, T_K, T_snow, arr, dose, m_new, w_new, w_old, phi, w, xi, e
  real :: rho, Ks, mass, vol, kh
  logical :: fresh
  integer :: i, j, k, ctr, nig

  dt_days = CS%t_accum / 86400.0
  I_t = 1.0 / CS%t_accum
  phi = 0.0 ; if (CS%stochastic) phi = exp(-dt_days / CS%st_tau_f)
  ctr = int(time_type_to_real(CS%Time) / 86400.0)     ! day counter for the random streams
  nig = G%ieg - G%isg + 1

  !$OMP parallel do default(shared) private(U, T_K, T_snow, arr, dose, m_new, w_new, w_old, w, xi, &
  !$OMP                                     e, rho, Ks, mass, vol, kh, fresh)
  do j=G%jsc,G%jec ; do i=G%isc,G%iec
    U = CS%U_sum(i,j) * I_t
    if (CS%vw_ema7(i,j) <= 0.0) then   ! first day, or not found in the restart file
      CS%vw_ema7(i,j) = U
    else
      CS%vw_ema7(i,j) = CS%vw_ema7(i,j) + (U - CS%vw_ema7(i,j)) * min(dt_days / CS%tau_ema, 1.0)
    endif
    dose = max(U - CS%U_pack, 0.0)**3 * dt_days

    do k=1,IG%CatIce
      m_new = US%RZ_to_kg_m2 * IST%mH_snow(i,j,k)
      if (m_new <= 0.0) then
        CS%tr(i,j,k,1,:) = 0.0 ; CS%rho_cat(i,j,k) = 0.0 ; CS%Ks_cat(i,j,k) = 0.0
        if (CS%stochastic) CS%eta(i,j,k,1,:) = 0.0
        cycle
      endif

      T_K = CS%Ts_sum(i,j,k) * I_t
      arr = exp(CS%Q_over_R * (1.0/CS%T_ref - 1.0/min(T_K, 273.15))) * dt_days
      ! A fresh pack: no snow at the start of the day, and none carried in by transport.
      fresh = (CS%m_start(i,j,k) <= 0.0) .and. (CS%tr(i,j,k,1,I_AGE) <= 0.0)
      if (fresh) then        ! new snow enters with half a day of exposure
        CS%tr(i,j,k,1,I_AGE)  = 0.5*arr
        CS%tr(i,j,k,1,I_DOSE) = 0.5*dose
        CS%tr(i,j,k,1,I_DEP)  = U
      else
        w_new = min(CS%snow_sum(i,j,k), m_new) / m_new
        w_old = 1.0 - w_new
        CS%tr(i,j,k,1,I_AGE)  = (CS%tr(i,j,k,1,I_AGE)  + arr)  * w_old + 0.5*arr  * w_new
        CS%tr(i,j,k,1,I_DOSE) = (CS%tr(i,j,k,1,I_DOSE) + dose) * w_old + 0.5*dose * w_new
        CS%tr(i,j,k,1,I_DEP)  = CS%tr(i,j,k,1,I_DEP) * w_old + U * w_new
      endif

      e = 0.0
      if (CS%stochastic) then
        if (fresh) then
          CS%eta(i,j,k,1,I_ETA_P) = gauss(CS, ctr, i + G%idg_offset, j + G%jdg_offset, nig, k, 1)
          CS%eta(i,j,k,1,I_ETA_F) = gauss(CS, ctr, i + G%idg_offset, j + G%jdg_offset, nig, k, 2)
        else
          xi = gauss(CS, ctr, i + G%idg_offset, j + G%jdg_offset, nig, k, 2)
          CS%eta(i,j,k,1,I_ETA_F) = phi*CS%eta(i,j,k,1,I_ETA_F) + sqrt(1.0 - phi**2) * xi
          w = min(0.999, CS%st_beta * min(CS%snow_sum(i,j,k), m_new) / m_new)
          if (w > 0.0) then
            xi = gauss(CS, ctr, i + G%idg_offset, j + G%jdg_offset, nig, k, 1)
            CS%eta(i,j,k,1,I_ETA_P) = ((1.0 - w)*CS%eta(i,j,k,1,I_ETA_P) + w*xi) / &
                                      sqrt((1.0 - w)**2 + w**2)
          endif
        endif
        e = error_from_eta(CS, sqrt(CS%st_a)*CS%eta(i,j,k,1,I_ETA_P) + &
                               sqrt(1.0 - CS%st_a)*CS%eta(i,j,k,1,I_ETA_F))
      endif

      if (CS%Tsn_t(i,j,k) > 0.0) then
        T_snow = CS%Tsn_sum(i,j,k) / CS%Tsn_t(i,j,k)
      else
        T_snow = US%C_to_degC*Temp_from_En_S(IST%enth_snow(i,j,k,1), 0.0, IST%ITV)
      endif
      call snow_rho_Ks(CS, m_new, CS%tr(i,j,k,1,I_AGE), CS%tr(i,j,k,1,I_DOSE), CS%tr(i,j,k,1,I_DEP), &
                       CS%vw_ema7(i,j), e, T_snow, CS%rho_cat(i,j,k), CS%Ks_cat(i,j,k))
    enddo

    ! Cell means for the 2-d SNOW_PROPERTIES_2D fields: density is total mass over total volume, so
    ! that m/rho summed over categories gives the right snow thickness; conductivity is thickness
    ! weighted.  Where there is no snow, use fresh snow under today's wind at -10 degC.
    if (allocated(IST%rho_snow)) then
      mass = 0.0 ; vol = 0.0 ; kh = 0.0
      do k=1,IG%CatIce ; if ((IST%mH_snow(i,j,k) > 0.0) .and. (IST%part_size(i,j,k) > 0.0)) then
        mass = mass + IST%part_size(i,j,k) * IST%mH_snow(i,j,k)
        vol  = vol  + IST%part_size(i,j,k) * IST%mH_snow(i,j,k) / CS%rho_cat(i,j,k)
        kh   = kh   + IST%part_size(i,j,k) * IST%mH_snow(i,j,k) / CS%rho_cat(i,j,k) * CS%Ks_cat(i,j,k)
      endif ; enddo
      if (vol > 0.0) then
        rho = mass / vol ; Ks = kh / vol
      else
        call snow_rho_Ks(CS, 0.0, 0.0, 0.5*dose, U, CS%vw_ema7(i,j), 0.0, -10.0, rho, Ks)
      endif
      IST%rho_snow(i,j) = US%kg_m3_to_R * rho
      if (allocated(IST%Ks_snow)) &
        IST%Ks_snow(i,j) = Ks * US%W_m2_to_QRZ_T * US%m_to_Z * US%C_to_degC
    endif

    ! Reset the accumulators for the next day.
    do k=1,IG%CatIce
      CS%m_start(i,j,k) = US%RZ_to_kg_m2 * IST%mH_snow(i,j,k)
      CS%Ts_sum(i,j,k) = 0.0 ; CS%Tsn_sum(i,j,k) = 0.0 ; CS%Tsn_t(i,j,k) = 0.0
      CS%snow_sum(i,j,k) = 0.0
    enddo
    CS%U_sum(i,j) = 0.0
  enddo ; enddo
  CS%t_accum = 0.0
end subroutine snow_pack_daily_update


!> Density and conductivity in SI units from the pack properties.
subroutine snow_rho_Ks(CS, SWE, age, dose, dep, ema, e, T_snow, rho, Ks)
  type(snow_pack_CS), intent(in)  :: CS     !< This package's control structure
  real,               intent(in)  :: SWE    !< Snow mass per unit ice area [kg m-2]
  real,               intent(in)  :: age    !< Arrhenius-weighted age [days]
  real,               intent(in)  :: dose   !< Wind-packing exposure [m3 s-3 days]
  real,               intent(in)  :: dep    !< Wind at deposition [m s-1]
  real,               intent(in)  :: ema    !< 7-day running-mean wind [m s-1]
  real,               intent(in)  :: e      !< Stochastic error added to z [nondim]
  real,               intent(in)  :: T_snow !< Snow temperature [degC]
  real,               intent(out) :: rho    !< Snow density [kg m-3]
  real,               intent(out) :: Ks     !< Snow conductivity [W m-1 K-1]
  real :: z, r

  z = CS%b0 - exp(CS%c_thin) / (1.0 + max(SWE, 0.0)) + CS%w_age * log(1.0 + age) + &
      CS%w_wind * (log(1.0 + dose) + dep + ema) + e
  rho = min(max(exp(CS%log_rho_max) / (1.0 + exp(-z)), CS%rho_min), CS%rho_max)

  if (trim(CS%Ks_scheme) == "SNOWPACK") then
    Ks = exp((rho/300.0)**CS%Ks_p - CS%Ks_c + CS%Ks_gT * min(T_snow, 0.0))
  elseif (trim(CS%Ks_scheme) == "STURM97") then   ! Sturm et al. (1997), density in g cm-3
    r = 1.0e-3 * rho
    if (r < 0.156) then
      Ks = 0.023 + 0.234*r
    else
      Ks = 0.138 - 1.01*r + 3.233*r**2
    endif
  else
    Ks = CS%Ks_const
  endif
end subroutine snow_rho_Ks


!> The error in z for a standardised value eta, from the quantile table, with eta clamped.
real function error_from_eta(CS, eta)
  type(snow_pack_CS), intent(in) :: CS  !< This package's control structure
  real,               intent(in) :: eta !< Standardised error [nondim]
  real :: x, f
  integer :: n

  x = min(max(eta, -CS%st_eta_max), CS%st_eta_max)
  x = min(max(x, ETA_TAB(1)), ETA_TAB(NQ))
  n = min(int((x - ETA_TAB(1)) / (ETA_TAB(2) - ETA_TAB(1))) + 1, NQ - 1)
  f = (x - ETA_TAB(n)) / (ETA_TAB(n+1) - ETA_TAB(n))
  error_from_eta = (1.0 - f)*E_TAB(n) + f*E_TAB(n+1)
end function error_from_eta


!> A standard-normal random number for one cell, category, day and stream, by Box-Muller from a
!! counter-based generator, so the value depends only on its arguments and the seed.
real function gauss(CS, ctr, ig, jg, nig, k, stream)
  type(snow_pack_CS), intent(in) :: CS     !< This package's control structure
  integer,            intent(in) :: ctr    !< Day counter
  integer,            intent(in) :: ig, jg !< Global cell indices
  integer,            intent(in) :: nig    !< Global number of cells in i
  integer,            intent(in) :: k      !< Thickness category
  integer,            intent(in) :: stream !< 1 = persistent part, 2 = fast part
  real, parameter :: TWO_PI = 6.283185307179586
  real :: u1, u2
  integer :: key

  ! The key identifies the stream (cell, category, eta_p or eta_f).  The seed shifts the counter by
  ! SEED_STRIDE days, so each seed uses its own block of counter values and ensemble members are
  ! independent; combining the seed with the key instead would only relabel the streams.
  integer, parameter :: SEED_STRIDE = 1000000
  integer :: c

  key = (((jg - 1)*nig + (ig - 1))*16 + (k - 1))*2 + (stream - 1)
  c = 2*(ctr + CS%st_seed*SEED_STRIDE)
  u1 = max(random_01_CB(c, key), 1.0e-12)
  u2 = random_01_CB(c + 1, key)
  gauss = sqrt(-2.0*log(u1)) * cos(TWO_PI*u2)
end function gauss


!> Post the tracer, density and conductivity diagnostics (call with averaging enabled).
subroutine snow_pack_post_diags(CS)
  type(snow_pack_CS), pointer :: CS !< This package's control structure
  integer :: m

  if (.not.associated(CS)) return
  do m=1,NTR
    if (CS%id_tr(m) > 0) call post_data(CS%id_tr(m), CS%tr(:,:,:,1,m), CS%diag)
  enddo
  if (CS%id_ema > 0) call post_data(CS%id_ema, CS%vw_ema7, CS%diag)
  if (CS%id_rho > 0) call post_data(CS%id_rho, CS%rho_cat, CS%diag)
  if (CS%id_Ks > 0)  call post_data(CS%id_Ks,  CS%Ks_cat,  CS%diag)
  if (CS%stochastic) then
    do m=1,2
      if (CS%id_eta(m) > 0) call post_data(CS%id_eta(m), CS%eta(:,:,:,1,m), CS%diag)
    enddo
  endif
end subroutine snow_pack_post_diags


!> Deallocate the package's memory.
subroutine snow_pack_tracers_end(CS)
  type(snow_pack_CS), pointer :: CS !< This package's control structure
  if (.not.associated(CS)) return
  if (associated(CS%tr)) deallocate(CS%tr)
  if (associated(CS%eta)) deallocate(CS%eta)
  deallocate(CS)
end subroutine snow_pack_tracers_end

end module snow_pack_tracers
