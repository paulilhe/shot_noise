subroutine matrix_op(grid_x,grid_t,eval_grid,pas,K,L,M,C)
  COMPLEX :: i=(0.0,1.0)
  INTEGER , intent(in) :: K,L,M
  REAL , intent(in) :: pas
  INTEGER :: u,v,w
  REAL , dimension(1:M) , intent(in) :: grid_x
  REAL , dimension(1:K) , intent(in) :: grid_t
  REAL , dimension(1:L) , intent(in) :: eval_grid
  COMPLEX, dimension(1:L,1:M) , intent(out) :: C
  
  do u=1,L
     do v=1,M
        do w=1,K
           C(u,v) = C(u,v) - i*pas*grid_t(w)*grid_x(v)*exp(-i*grid_t(w)*grid_x(v)*eval_grid(u))
        end do
     end do
  end do
  
end subroutine matrix_op
