! ----------------------------------------------------------------------
! ----------------------------------------------------------------------
! FitzHugh-Nagumo computation of Isochron
! ----------------------------------------------------------------------
! Continuation of canard orbits in parameter space
! ----------------------------------------------------------------------
! ----------------------------------------------------------------------

      SUBROUTINE RHS(NDIM,U,PAR,F,JAC,A)
        IMPLICIT NONE
        INTEGER, INTENT(IN) :: NDIM
        DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
        LOGICAL, INTENT(IN) :: JAC
        DOUBLE PRECISION, INTENT(OUT) :: F(NDIM), A(NDIM,NDIM)
        DOUBLE PRECISION ap,bp,cp,zp
        
        ! Define the state variables
        ap = PAR(1)
        bp = PAR(2)
        cp = PAR(3)
        zp = PAR(4)
        
        F(1)  =  cp*(U(2) + U(1) - U(1)**3/3.0 + zp)
        F(2)  = -(U(1) -ap + bp*U(2) )/cp

        IF(JAC)THEN
           A(1,1) =   cp*(1.0 - U(1)**2)
           A(1,2) =   cp

           A(2,1) = -1.0/cp
           A(2,2) = -bp/cp
        ENDIF
      END SUBROUTINE RHS

      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
!     ---------- ----
        IMPLICIT NONE
        INTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC
        DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
        DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
        DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)
        DOUBLE PRECISION T, A(2,2)
        CALL RHS(2,U(1:2),PAR,F(1:2),.TRUE.,A)
        IF(NDIM==2) RETURN
        F(1:2)  =  PAR(11)*F(1:2)
        F(3:4)  =  PAR(11)*(MATMUL(TRANSPOSE(A),U(3:4)))
        IF(NDIM==4) RETURN
        CALL RHS(2,U(5:6),PAR,F(5:6),.FALSE.,A)
        F(5:6)  = PAR(12)*F(5:6)
        IF(NDIM==6) RETURN
        CALL RHS(2,U(7:8),PAR,F(7:8),.FALSE.,A)
        F(7:8)  = PAR(13)*F(7:8)
      END SUBROUTINE FUNC

! ----------------------------------------------------------------------
! ----------------------------------------------------------------------

      SUBROUTINE STPNT
      END SUBROUTINE STPNT

! ----------------------------------------------------------------------
! ----------------------------------------------------------------------
      SUBROUTINE BCND(NDIM,PAR,ICP,NBC,U0,U1,FB,IJAC,DBC) 
!     ---------- ---- 
        IMPLICIT NONE
        INTEGER, INTENT(IN) :: NDIM, ICP(*), NBC, IJAC
        DOUBLE PRECISION, INTENT(IN) :: PAR(*), U0(NDIM), U1(NDIM)
        DOUBLE PRECISION, INTENT(OUT) :: FB(NBC)
        DOUBLE PRECISION, INTENT(INOUT) :: DBC(NBC,*)
        ! Define boundary conditions */
        IF (NDIM==2) RETURN
        FB(1:2) = U0(1:2)-U1(1:2)
        FB(3:4) = U1(3:4)-PAR(34)*U0(3:4)
        FB(5)   = DOT_PRODUCT(U0(3:4),U0(3:4)) - PAR(35)
        IF (NDIM==4) RETURN
        FB(6:7) = U1(5:6) - (U0(1:2) + PAR(36)*(/ -U0(4), U0(3) /))
        IF (NDIM==6) RETURN
        FB(8:9)  = U1(7:8) - U0(1:2)
        FB(10:11)= U0(7:8) - PAR(70:71)
        FB(12)   = PAR(13) - PAR(14)*PAR(11)
      END SUBROUTINE BCND

! ----------------------------------------------------------------------
! ----------------------------------------------------------------------
      SUBROUTINE PVLS(NDIM,U,PAR)
!     ---------- ----
        IMPLICIT NONE
        INTEGER, INTENT(IN) :: NDIM
        DOUBLE PRECISION, INTENT(IN) :: U(NDIM)
        DOUBLE PRECISION, INTENT(INOUT) :: PAR(*)

        DOUBLE PRECISION, EXTERNAL :: GETP,GETU2
        INTEGER NDX,NCOL,NTST
        PAR(80)= GETP('STA',0,U)
        IF (NDIM==4) THEN
           PAR(70:71) = (/GETP("BV0",1,U), GETP("BV0",2,U)/)
        ENDIF
        IF (NDIM==8) THEN
           PAR(72:73) = (/GETP("BV0",7,U), GETP("BV0",8,U)/)
        ENDIF
      END SUBROUTINE PVLS
! ----------------------------------------------------------------------
! ----------------------------------------------------------------------
      SUBROUTINE ICND(NDIM,PAR,ICP,NINT,U,UOLD,UDOT,UPOLD,FI,IJAC,DINT)
!--------- ----
        IMPLICIT NONE
        INTEGER, INTENT(IN) :: NDIM, ICP(*), NINT, IJAC
        DOUBLE PRECISION, INTENT(IN) :: PAR(*)
        DOUBLE PRECISION, INTENT(IN) :: U(NDIM), UOLD(NDIM), UDOT(NDIM), UPOLD(NDIM)
        DOUBLE PRECISION, INTENT(OUT) :: FI(NINT)
        DOUBLE PRECISION, INTENT(INOUT) :: DINT(NINT,*)
        IF (NINT == 1) THEN
           FI(1) =  DOT_PRODUCT(U(1:2), UPOLD(1:2))
        ENDIF
      END SUBROUTINE ICND
! ----------------------------------------------------------------------
! ----------------------------------------------------------------------
      SUBROUTINE FOPT
      END SUBROUTINE FOPT
