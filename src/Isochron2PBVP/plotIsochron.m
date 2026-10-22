clc
clear all

strProb = 'fncMultiSeg';
angles  = 2*2*pi/90;
T       = [cos(angles), -sin(angles);sin(angles), cos(angles)];
cMap    = cool(100);
figure(1); hold on;
for i=1:100
    p                 =  load(['./' strProb '/dataIso/' strProb '_Iso_' num2str(i) '.csv']); 
    transCurve        =  (T*p(:,3:4)')';
    plot(transCurve(:,1),transCurve(:,2),'LineWidth',2,'Color',cMap(i,:))
end
perOrbit  = load(['./' strProb '/dataIso/' strProb '_periodicSolution']);
perOrbitT = (T*perOrbit(:,2:3)')';
plot(perOrbitT(:,1),perOrbitT(:,2),'LineWidth',1,'Color',[0,0,0])
axis([-2,2,-0.4,1.5])
%axis([0.83,1.03,-0.23,-0.15])