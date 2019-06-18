clear all
close all

%lattice of dielectric scatterers
rc = 1.6; 
yp0 =45;      %45;
xp0 =-16;      %-16;
ln = linspace(0,32,2);
[xp,yp] = ndgrid(ln,ln);

%xp = 0; %0 10 10]; % -5 25]  %[-20 0];% 20 -20 0 20 ]; % 3 -3 3 ]
%yp= 0;  %10 0 10]; % 10 30];  %[-3 -3]; %-3 6 6 6]; %, -20 -26 -26]

xp = xp + xp0; 
yp = yp + yp0;

%directization points on the surface of a circle
N = 50;

%objective region
x0c = 0.0;
y0c = 40+23;
rj = 60;
thn = 0:0.1:2*pi;

%location of the source
ys = 13; xs = -25:25;  src= xs + 1i * ys; %-30:30;

nc=length(xp(:));
%break
Nt=N*nc;

tol=1.0;
targold=0.0; targ=0.0;
trestold=0.0; trest=0.0;
count=0;
fact_x = 1e-2;
fact_y =1e-2;
dt = [];
dty = dt;
dt_x =1; %[1, 1, 1, 1];
dt_y = 1;
ttarg = [];
thrsh = 1;
abc = [];
while ( (thrsh > 1e-8) && (count < 1000) )
    
    figure(1)
 
    for ii=1:nc
        GG{ii} = cylinder(rc,xp(ii),yp(ii));
        GG{ii} = curvquad(GG{ii},'ptr',N,10);
       quiver(real(GG{ii}.x),imag(GG{ii}.x),real(GG{ii}.nx),imag(GG{ii}.nx));
        hold on
        plot(real(GG{ii}.x),imag(GG{ii}.x),'k','LineWidth',2)
        hold on
    end

    plot(x0c+rj*sin(thn),y0c+rj*cos(thn),'k','LineWidth',4)
    %in thick red the objective region
    plot(x0c+rj*sin(pi/6:0.1:pi/3),y0c+rj*cos(pi/6:0.1:pi/3),'r','LineWidth',6) %remets pi/3
    %source of incident wave
    plot(xs,ys,'ks','MarkerSize',8,'LineWidth',2)
    hold off
    axis equal
     %pause
    %uinc=exp(1i*k*abs(G.x-src))./abs(G.x-src)*1/4/pi;

    k = 2*pi/8; eta = k;                           % wavenumber, SLP mixing amount 
    f =  @(z) sum(1i*1*besselh(0,1,k*abs(z-src))/4.0,2);   % known soln: interior source
    fgradx = @(z) sum(-1i/4.0*1*k*besselh(1,1,k*abs(z-src)).*(real(z-src))./abs(z-src),2);
    fgrady = @(z) sum(-1i/4.0*1*k*besselh(1,1,k*abs(z-src)).*(imag(z-src))./abs(z-src),2);


    unic=[]; rhs=[]; 
    for ii=1:nc
        rhs = [rhs -2*f(GG{ii}.x)];
    end
    
    A = nan(N*nc,N*nc);
    for ii=1:nc
        for jj=1:nc
            for i=1:N
                for j=1:N
                    A(i+(ii-1)*N,j+(jj-1)*N) = 2*CFIEnystKR_src(GG{ii},GG{jj},i,j,k,eta);
                end
            end
        end
    end

    sigma =( A) \ rhs(:);
    dFdq =  A;
    
    %targetarea: obj=x0c+rj*sin(thn)+1i*(y0c+rj*cos(thn));
    tobj=pi/6:0.02:pi/3; %remets pi/3
    trest=[0:0.02:pi/6 pi/3:0.02:2*pi];
    obj=x0c+rj*sin(tobj)+1i*(y0c+rj*cos(tobj));
    objrest=x0c+rj*sin(trest)+1i*(y0c+rj*cos(trest));
    nobj=rj*(1i*sin(tobj)-cos(tobj));
    
    xcirc=xp+1i*yp;

    
    targold=targ;
    uobj = zeros(length(obj), nc);
    %uobjj = zeros(length(obj));
    for ii=1:nc
        for jj=1:length(obj)
                %d  = [d; obj(jj)-GG{ii}.x];  % displacement of targets from jth src pt        
            uobj(jj,ii)= evalCFIEhelm_src(obj(jj),GG{ii},sigma(1+(ii-1)*N:ii*N),k,eta);         % evaluate soln               
        end 
        
    end
    targ = -sum((abs(sum(uobj,2))).^2);
    ttarg =[ttarg, targ];   
       
       
    

    
    %derivative of J wrt \sigma 
    B_r = zeros(length(obj), N*nc);
    B_i = B_r;
    uobjj = zeros(length(obj),1);
    for ii=1:nc
        for jj=1:length(obj)
            uobjj(jj) = sum(uobj(jj,:));
            for i=1:N
                a_r =CFIEnystKR_src_derJq(obj,GG{ii},jj,i,k,eta);
                a_i = 1i*a_r;
                B_r(jj,i+(ii-1)*N) = 2 * real(conj(uobjj(jj)) * a_r);
                B_i(jj,i+(ii-1)*N) = 2 * real(conj(uobjj(jj)) * a_i);
            end
        end
    end
    dJdq_r = sum(B_r,1);
    dJdq_i = sum(B_i,1);
    AA = [real(dFdq),-imag(dFdq); imag(dFdq), real(dFdq)];
    dJdq = [dJdq_r, dJdq_i];
    [lambda] = dJdq / AA;
   
   % pause

    
    dFdxc=[];
    for ii=1:nc
        pff=zeros(size(GG{ii}.x));
        pff_x = pff;
        pff_y = pff;
        for jj=1:nc
                if (ii ~= jj) 
                    [uu, vv] = evalCFIEhelm_src_derF_S(GG{ii}.x,GG{jj},sigma(1+(jj-1)*N:jj*N),k,eta);
                    pff_x = pff_x+2*uu;  
                    pff_y = pff_y+2*vv;
                    [uuu, vvv] = evalCFIEhelm_src_derF_S2(GG{jj}.x,GG{ii},sigma(1+(ii-1)*N:ii*N),k,eta);
                    pf_x(N*(jj-1)+1:N*jj,ii) = 2*uuu;
                    pf_y(N*(jj-1)+1:N*jj,ii)  = 2*vvv;
                end              
        end
        pf_x(N*(ii-1)+1:N*ii,ii) = pff_x+2*fgradx(GG{ii}.x);
        pf_y(N*(ii-1)+1:N*ii,ii) = pff_y+2*fgrady(GG{ii}.x);
    end
    dFdxc = pf_x;
    dFdyc = pf_y;    
    
    

    dJdxc=[];
    dJdyc=[];
    for ii=1:nc 
        for jj=1:length(obj)
           [uu, vv] = ((evalCFIEhelm_src_der_G_x(obj(jj),GG{ii},sigma(1+(ii-1)*N:ii*N),k,eta)));
            pj_x(jj,ii) = 2*real(conj(uobjj(jj))*uu);
            pj_y(jj,ii) = 2*real(conj(uobjj(jj))*vv);
        end
        dJdxc(ii) =  sum(pj_x(:,ii)); %-.3*( (xp(ii)-x0c) )./((xp(ii)-x0c).^2+(yp(ii)-y0c).^2-60^2);
        dJdyc(ii) =  sum(pj_y(:,ii)); %-.3*( (yp(ii)-y0c))./((xp(ii)-x0c).^2+(yp(ii)-y0c).^2 - 60^2);
%         for kk = 1:nc
%             if (kk ~= ii)
%                 dJdxc(ii) = dJdxc(ii) - .5*( -(xp(ii) - xp(kk) ) ) ./ ( (xp(ii) - xp(kk) ).^2+( yp(ii)-yp(kk) ).^2+(2*rc)^2+6);
%                 dJdyc(ii) =  dJdyc(ii) - .5 * ( -(yp(ii) - yp(kk) )) ./ ( (xp(ii) - xp(kk )).^2+( yp(ii)-yp(kk) ).^2+(2*rc)^2+6)-...
%                                    .6*(-1)/(yp(kk)+25);
%             end
%         end
    end
    
         
    dt_x_old = dt_x;
    dt_y_old = dt_y;
    dt_x = lambda*[(real(dFdxc));imag(dFdxc)] - dJdxc;
    dt_y = lambda*[real(dFdyc);imag(dFdyc)] - dJdyc;
   
    dt=[dt, dt_x];
    dty = [dty, dt_y];
   
    

    xp_old = xp;
    yp_old = yp;
    fact_x = 1e-2;
    fact_y = 1e-2;
    xp(:)=xp(:)-fact_x*dt_x';
    yp(:)=yp(:)-fact_y*dt_y';
    dx = xp-xp_old;
    dy =yp - yp_old;
    abc = [abc, xp];
    
  
   %fact_x = (abs((dx(:))'*(dt_x'-(dt_x_old)'))) /norm(dt_x'-(dt_x_old)')^2
   %fact_y = (abs((dy(:))'*(dt_y'-(dt_y_old)'))) / norm(dt_y'-(dt_y_old)')^2;
    count=count+1;
    tol(count) = (abs(targ-targold));
    %tol(count)=norm((xp-xp_old)+(yp-yp_old),2)
    thrsh = tol(count);
    
end
figure;
plot(-ttarg)


