      IMPLICIT REAL*8 (A-H, L-Z)
      real r,lg,lb,lh,r1(10000)
      real N1,N
      integer I,J,K
      OPEN(14,file='M2',status='unknown')
      open(100,file='M2.dat',status='unknown')

      r1(0) = 0.0

      do I = 1, 2677
      read(14,*)r,lg,lb
       r1(i) = r

       if(r.ne.r1(i-1))then
       n=n+1
       write(100,10)r,lg,lb


       endif

       enddo

       if(n.le.5000)then
       do k=int(n),4999
       write(100,10)r,lg,lb
       enddo
       endif
       print*,n,k


10    FORMAT(4f12.4,1f12.5,2f12.2,1E12.3,1f12.2,4i6,2f12.4,2E14.6)
       end
