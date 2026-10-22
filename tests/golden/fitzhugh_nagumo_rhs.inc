        F(1) = PAR(3)*(PAR(4) - 1.0d0/3.0d0*U(1)**3 + U(1) + U(2))
        F(2) = (PAR(1) - PAR(2)*U(2) - U(1))/PAR(3)
        IF(JAC)THEN
           A(1,1) = PAR(3)*(1 - U(1)**2)
           A(1,2) = PAR(3)
           A(2,1) = -1/PAR(3)
           A(2,2) = -PAR(2)/PAR(3)
        ENDIF
