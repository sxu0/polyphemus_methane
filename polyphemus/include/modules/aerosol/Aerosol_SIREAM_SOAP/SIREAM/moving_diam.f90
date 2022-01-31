SUBROUTINE MOVING_DIAM(ns, nesp, eh2o, &
                       kloc, Qesp, N)

!!$------------------------------------------------------------------------
!!$     
!!$     -- DESCRIPTION 
!!$     
!!$     This subroutine redistribute the concentrations after the GDE.
!!$     Moving Diameter
!!$     
!!$------------------------------------------------------------------------
!!$     
!!$     -- INPUT VARIABLES
!!$     
!!$     
!!$     ns             : number of sections
!!$     nesp           : number of species
!!$     klock          : list of location of sections after condensation/evaporation
!!$ 
!!$     -- VARIABLES
!!$     
!!$     Q_tmp     : Temporary Mass concentration 
!!$     N_tmp     : Temporary number concentration 
!!$
!!$     -- INPUT/OUTPUT VARIABLES
!!$      
!!$     N              : Number concentration by bin
!!$     Qesp           : Mass concentration by bin and species
!!$      
!!$     -- OUTPUT VARIABLES
!!$     
!!$     
!!$------------------------------------------------------------------------



  IMPLICIT NONE

  INCLUDE '../INC/parameuler.inc'


!!! ------ Input 
  INTEGER, INTENT(in) :: ns, nesp
  INTEGER, DIMENSION(ns), INTENT(in) :: kloc
   integer eh2o

!!! ------ Input/Output 
  DOUBLE PRECISION, DIMENSION(ns), INTENT(inout) :: N
  DOUBLE PRECISION, DIMENSION(ns, nesp), INTENT(inout) :: Qesp

!!! ------  
  INTEGER k, jesp, s, i
  DOUBLE PRECISION, DIMENSION(ns) :: N_tmp
  DOUBLE PRECISION, DIMENSION(ns) :: Q  
  DOUBLE PRECISION, DIMENSION(ns, nesp) :: Q_tmp
!!! ~~~~~~ Distribution of number concentration by species
    do k=1,ns
      N_tmp(k) = 0.d0!transver vector of Number
      do s= 1, nesp      
	Q_tmp(k,s) = 0.d0!transver vector of Mass
      enddo
    enddo
    
    do k=1,ns
       N_tmp(kloc(k)) = N_tmp(kloc(k)) + N(k)!defined which bin it will located	  
       do s= 1, nesp      
	    Q_tmp(kloc(k),s) = Q_tmp(kloc(k),s) + Qesp(k,s)!transver vector of Mass
       enddo
    enddo

    !update each bin        
    do k=1,ns
      N(k)=N_tmp(k)!transver vector of Number
      do s= 1, nesp      
	Qesp(k,s)=Q_tmp(k,s)!transver vector of Mass	
      enddo
    enddo
    
  DO k = 1,ns !size bins
     Q(k) = 0.d0
     DO jesp = 1, nesp!species
        if (jesp .ne. eh2o) then
        Q(k) = Q(k) + Qesp(k, jesp)
        endif
     ENDDO
  ENDDO        
    
  CALL TEST_MASS_NB(ns,nesp-1,Q,N,Qesp)

END SUBROUTINE MOVING_DIAM