module Forpy_interface

use forpy_mod,                 only : module_py,list,ndarray,object,tuple
use forpy_mod,                 only : err_print
use forpy_mod,                 only : forpy_initialize,get_sys_path,import_py,print_py
use forpy_mod,                 only : ndarray_create,tuple_create,call_py,cast
use forpy_mod,                 only : forpy_finalize

implicit none; private

public :: forpy_run_python_init,forpy_run_python,forpy_run_python_finalize

!> Control structure for Python interface
type, public :: python_interface ; private
  type(module_py) :: pymodule
  type(list) :: paths
end type

contains

!> Initialize Forpy with specify Python script and directory
subroutine forpy_run_python_init(CS,python_dir,python_file)
    character(len=*),         intent(in)  :: python_dir   !< The directory in which python scripts are found
    character(len=*),         intent(in)  :: python_file  !< The name of the Python script to read
    type(python_interface),   intent(inout) :: CS         !< Python interface object
    ! Local Variables
    integer :: ierror ! return code from python interfaces
    ierror = forpy_initialize()
    write(*,*) "############ Initialize Forpy ############"
    ierror = get_sys_path(CS%paths)
    ierror = CS%paths%append(python_dir)
    ierror = import_py(CS%pymodule,python_file)
    if (ierror/=0) then; call err_print; endif
    ierror = print_py(CS%pymodule)
    if (ierror/=0) then; call err_print; endif
  
end subroutine forpy_run_python_init

!> !> Send variables to a python script and output the results
subroutine forpy_run_python(IN_CNN, IN_ANN, CNN_weights, ANN_weights, dsic_mu, dsic_std, dCN, CS)
    real, dimension(:,:,:), intent(in) :: IN_CNN       !< matrix of input states to CNN
    real, dimension(:,:,:), intent(in) :: IN_ANN       !< matrix of input states to ANN
    character(len=*),       intent(in) :: CNN_weights  !< filename of Torch weights for CNN
    character(len=*),       intent(in) :: ANN_weights  !< filename of Torch weights for ANN
    real,                   intent(in) :: dsic_mu      !< normalization mean for CNN output
    real,                   intent(in) :: dsic_std     !< normalization standard dev. for CNN output
    real, dimension(:,:,:), intent(inout) :: dCN       !< ANN predictions of dCN
    type(python_interface), intent(in) :: CS           !< Python interface object
    
    ! Local Variables for Forpy
    integer       :: ierror                     !< return code from python interfaces
    type(ndarray) :: in1_py,in2_py,out_arr      !< variables in the form of numpy array
    type(object)  :: obj                        !< return objects
    type(tuple)   :: args                       !< input arguments for the Python module
    real, dimension(:,:,:), pointer :: out_for  !< outputs from Python module
    integer :: hi, hj ! temporary
    integer :: i, j, k

    ! Covert input into Forpy Numpy Arrays 
    ierror = ndarray_create(in1_py, IN_CNN)
    if (ierror/=0) then; call err_print; endif
    ierror = ndarray_create(in2_py, IN_ANN)
    if (ierror/=0) then; call err_print; endif

    ! Create Python Argument 
    ierror = tuple_create(args,6)
    if (ierror/=0) then; call err_print; endif
    ierror = args%setitem(0,in1_py)
    ierror = args%setitem(1,in2_py)
    ierror = args%setitem(2,CNN_weights)
    ierror = args%setitem(3,ANN_weights)
    ierror = args%setitem(4,dsic_mu)
    ierror = args%setitem(5,dsic_std)

    if (ierror/=0) then; call err_print; endif
    
    ! Invoke Python 
    ierror = call_py(obj, CS%pymodule, "DAML", args)
    if (ierror/=0) then; call err_print; endif
    ierror = cast(out_arr, obj)
    if (ierror/=0) then; call err_print; endif
    ierror = out_arr%get_data(out_for, order='C')
    if (ierror/=0) then; call err_print; endif

    ! Destroy Objects
    call in1_py%destroy
    call in2_py%destroy   
    call out_arr%destroy
    call obj%destroy
    call args%destroy
    
    out1 = 0.0
    do i=1,size(out_for,3) ; do j=1,size(out_for,2) ; do k=1,size(out_for,1)
       out1(i,j,k) = out_for(k,j,i)
    enddo; enddo; enddo
  
end subroutine forpy_run_python 

!> Finalize Forpy
subroutine forpy_run_python_finalize(CS)
    type(python_interface), intent(inout) :: CS !< Python interface object
    write(*,*) "############ Finalize Forpy ############"
    call CS%pymodule%destroy
    call CS%paths%destroy
   
    call forpy_finalize
  
end subroutine forpy_run_python_finalize

end module Forpy_interface
