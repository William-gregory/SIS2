module Forpy_interface

!use MOM_coms,                  only : PE_here,num_PEs
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
subroutine forpy_run_python(in1, out1, CS, TopLayer)
    type(python_interface),        intent(in)  :: CS     !< Python interface object
  ! Local Variables
    logical, intent(in) :: TopLayer             !< If true, only top layer is used.
    real, dimension(:,:,:,:), &
                                    intent(in) :: in1     ! input variables.
    real, dimension(:,:,:,:), &
                                    intent(inout) :: out1      ! output variables.
  ! Local Variables for Forpy
    integer :: ierror ! return code from python interfaces
    type(ndarray) :: in1_py,out_arr         !< variables in the form of numpy array
    type(object)  :: obj                    !< return objects
    type(tuple)   :: args                   !< input arguments for the Python module
    real, dimension(:,:,:,:), pointer :: out_for  !< outputs from Python module
    !integer :: current_pe
    integer :: hi, hj ! temporary
    integer :: i, j, k, l
    integer :: nztemp, out_num

    !current_pe = PE_here()

    if (TopLayer) then
      nztemp = 1
    else
      nztemp = size(in1,4)
    endif

  ! Covert input into Forpy Numpy Arrays 
    if (TopLayer) then
        ierror = ndarray_create(in1_py, in1(:,:,:,1))
      else
        ierror = ndarray_create(in1_py, in1)
      endif
      if (ierror/=0) then; call err_print; endif
    
      ! Create Python Argument 
      ierror = tuple_create(args,3)
      if (ierror/=0) then; call err_print; endif
      ierror = args%setitem(0,in1_py)
      !ierror = args%setitem(1,current_pe)
      !ierror = args%setitem(2,num_PEs())
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
      call out_arr%destroy
      call obj%destroy
      call args%destroy

      ! find the margin size (if order='C')
      hi = (size(out1,2) - size(out_for,3))/2
      hj = (size(out1,3) - size(out_for,2))/2
    
      ! Output (out_for in C order has index (nk,nj,ni))
                      ! in F order has index (ni,nj,nk)
      out1 = 0.0
      do k=1,nztemp
        do j=1,size(out_for,2) ; do i=1,size(out_for,3); do l=1,size(out_for,4)
          out1(l,i+hi,j+hj,k) = out_for(k,j,i,l) ! if order='C'
          ! out1(l,i+hi,j+hj,k) = out_for(l,i,j,k) ! if order='F'
        enddo ; enddo ; enddo
      enddo
  
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
