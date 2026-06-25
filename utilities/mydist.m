function [res] = mydist()
d =[];
theta = [];
obj = findobj(gca,'tag','myline');

if ~isempty(obj)
    delete(obj)
    pause(.5)
end

roi = drawline(gca,'tag','myline');
x1 = (roi.Position(1,1));
x2 = (roi.Position(2,1));

y1 = (roi.Position(1,2));
y2 = (roi.Position(2,2));


delta = abs(diff(roi.Position));
d = norm(delta);
theta1 = 180*atan(delta(2)./delta(1))/pi;
theta2 = 180*atan(delta(1)./delta(2))/pi;
res.theta(1) = theta1;
res.theta(2) = theta2;
res.dx = delta(1);
res.dy = delta(2);
res.dist = d;
res.roi = roi;

hold on
plot([x1 x2],[y2 y2],'--r','tag','myline')
plot([x1 x1],[y1 y2],'--m','tag','myline')
plot([x1 x2],[y1 y2],'--c','tag','myline')

text(mean([x1 x2]),y2,'a','tag','myline','FontSize',12)
text(x1,mean([y1 y2]),'b','tag','myline','FontSize',12)
text(mean([x1 x2]),mean([y1 y2]),'c','tag','myline','FontSize',12)
legend({sprintf('a = %0.3f',delta(1)),...
        sprintf('b = %0.3f',delta(2)),...
        sprintf('c = %0.3f',d)})
title(sprintf('theta = %0.2f^o',min([theta1 theta2])))
hold off



