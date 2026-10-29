! ----------------------------------------------------------------------
! Isochrons of a planar limit cycle by continuation of a multi-segment
! two-point boundary value problem in AUTO-07p (Langfield, Krauskopf &
! Osinga 2014 for the method; the multi-segment set-up follows Hannam,
! Krauskopf & Osinga's CoCo implementation). Generic over the system:
! the vector field and its Jacobian are INCLUDEd from rhs.inc, the
! parameter indices from indices.inc, both written by
! deep_isochron.analysis.ode.auto.codegen next to this file (ADR-0016).
!
! Staged extended system, selected by NDIM:
!   2   the vector field itself (equilibria, periodic orbits)
!   4   two orbit segments U(1:2), U(3:4) with times IP_SEG1, IP_SEG2
!       joined into one periodic orbit (period IP_PERIOD = IP_SEG1 + IP_SEG2),
!       segment 1 starting on the section through IP_SECTION_POINT_X/Y with
!       normal IP_SECTION_NORMAL_X/Y: the cycle's phase 0
!   8   plus the adjoint equation w' = -A^T w on both segments (U(5:6),
!       U(7:8)); at the branch point IP_MULT = 1, w is the direction of the
!       phase sensitivity Z and its perpendicular the stable direction
!       (docs/design/isochrons.md, section 3)
!  10   plus an orbit segment U(9:10) of time IP_COPIES ending on the stable
!       direction at the junction, at distance IP_ETA: its start traces
!       the isochron (IP_POINT_X, IP_POINT_Y) of the junction (IP_BASE_*)
! ----------------------------------------------------------------------

      SUBROUTINE RHS(NDIM,U,PAR,F,JAC,A)
        IMPLICIT NONE
        INTEGER, INTENT(IN) :: NDIM
        DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
        LOGICAL, INTENT(IN) :: JAC
        DOUBLE PRECISION, INTENT(OUT) :: F(NDIM), A(NDIM,NDIM)
        INCLUDE 'rhs.inc'
      END SUBROUTINE RHS

      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
        IMPLICIT NONE
        INTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC
        DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
        DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
        DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)
        DOUBLE PRECISION A(2,2),B(2,2)
        INCLUDE 'indices.inc'
        CALL RHS(2,U(1:2),PAR,F(1:2),.TRUE.,A)
        IF(NDIM==2) RETURN
        F(1:2)  =  PAR(IP_SEG1)*F(1:2)
        CALL RHS(2,U(3:4),PAR,F(3:4),.TRUE.,B)
        F(3:4)  =  PAR(IP_SEG2)*F(3:4)
        IF(NDIM==4) RETURN
        F(5:6)  = -PAR(IP_SEG1)*(MATMUL(TRANSPOSE(A),U(5:6)))
        F(7:8)  = -PAR(IP_SEG2)*(MATMUL(TRANSPOSE(B),U(7:8)))
        IF(NDIM==8) RETURN
        CALL RHS(2,U(9:10),PAR,F(9:10),.FALSE.,A)
        F(9:10)  = PAR(IP_COPIES)*F(9:10)
      END SUBROUTINE FUNC

      SUBROUTINE STPNT(NDIM,U,PAR,T)
        ! every run starts from a solution passed in from Python
        IMPLICIT NONE
        INTEGER, INTENT(IN) :: NDIM
        DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
        DOUBLE PRECISION, INTENT(IN) :: T
      END SUBROUTINE STPNT

      SUBROUTINE BCND(NDIM,PAR,ICP,NBC,U0,U1,FB,IJAC,DBC)
        IMPLICIT NONE
        INTEGER, INTENT(IN) :: NDIM, ICP(*), NBC, IJAC
        DOUBLE PRECISION, INTENT(IN) :: PAR(*), U0(NDIM), U1(NDIM)
        DOUBLE PRECISION, INTENT(OUT) :: FB(NBC)
        DOUBLE PRECISION, INTENT(INOUT) :: DBC(NBC,*)
        INCLUDE 'indices.inc'
        IF (NDIM==2) RETURN
        ! the two segments close up into one periodic orbit
        FB(1:2)  = U0(1:2)-U1(3:4)
        FB(3:4)  = U1(1:2)-U0(3:4)
        FB(5)    = PAR(IP_PERIOD)-PAR(IP_SEG1)-PAR(IP_SEG2)
        ! phase 0: the start of segment 1 lies on the section (p, n)
        FB(6)    = DOT_PRODUCT(U0(1:2)-PAR(IP_SECTION_POINT_X:IP_SECTION_POINT_Y), &
                   PAR(IP_SECTION_NORMAL_X:IP_SECTION_NORMAL_Y))
        IF (NDIM==4) RETURN
        ! w' = -A^T w: periodic up to the multiplier, normalized at the junction
        FB(7:8)  = U1(7:8)- U0(5:6)
        FB(9:10) = U1(5:6)- PAR(IP_MULT)*U0(7:8)
        FB(11)   = DOT_PRODUCT(U0(7:8),U0(7:8)) - PAR(IP_NORM)
        IF (NDIM==8) RETURN
        ! the isochron segment ends on the line through the junction U0(3:4)
        ! perpendicular to w, i.e. along the stable direction, at distance eta
        FB(12:13)= U1(9:10)- (U0(3:4) + PAR(IP_ETA)*(/ -U0(8), U0(7) /))
      END SUBROUTINE BCND

      SUBROUTINE PVLS(NDIM,U,PAR)
        IMPLICIT NONE
        INTEGER, INTENT(IN) :: NDIM
        DOUBLE PRECISION, INTENT(IN) :: U(NDIM)
        DOUBLE PRECISION, INTENT(INOUT) :: PAR(*)
        DOUBLE PRECISION, EXTERNAL :: GETP
        INCLUDE 'indices.inc'
        IF (NDIM==10) THEN
           PAR(IP_BASE_X:IP_BASE_Y) = (/GETP("BV0",3,U), GETP("BV0",4,U)/)
           PAR(IP_POINT_X:IP_POINT_Y) = (/GETP("BV0",9,U), GETP("BV0",10,U)/)
        ENDIF
      END SUBROUTINE PVLS

      SUBROUTINE ICND(NDIM,PAR,ICP,NINT,U,UOLD,UDOT,UPOLD,FI,IJAC,DINT)
        IMPLICIT NONE
        INTEGER, INTENT(IN) :: NDIM, ICP(*), NINT, IJAC
        DOUBLE PRECISION, INTENT(IN) :: PAR(*)
        DOUBLE PRECISION, INTENT(IN) :: U(NDIM), UOLD(NDIM), UDOT(NDIM), UPOLD(NDIM)
        DOUBLE PRECISION, INTENT(OUT) :: FI(NINT)
        DOUBLE PRECISION, INTENT(INOUT) :: DINT(NINT,*)
        IF (NINT == 1) THEN
           FI(1) =  DOT_PRODUCT(U(1:2), UPOLD(1:2))+DOT_PRODUCT(U(3:4), UPOLD(3:4))
        ENDIF
      END SUBROUTINE ICND

      SUBROUTINE FOPT
      END SUBROUTINE FOPT
