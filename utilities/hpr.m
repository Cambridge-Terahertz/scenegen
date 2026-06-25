
 function visiblePtInds = hpr(p,C,param)
 % code taken fron: katz_tal_basri_-_direct_visibility_of_point_sets.pdf
 
 % p = point cloud data
 % C = camera position
 % param is HPR culling factor
 

 
 % Movethe points s.t. C is the origin
 % p=p-repmat(C,[numPts 1]);
 p=p-C;
 
 %Calculate ||p||
 normp=sqrt(dot(p,p,2));
 
 %Sphere radius
 % R=repmat(max(normp)*(10^param),[numPts 1]);
 R=max(normp)*(10^param);
 
 %Spherical flipping
 % P=p+2*repmat(R-normp,[1 dim]).*p./repmat(normp,[1 dim]);

 P=p+2*(R-normp).*p./normp;
 
 %convex hull
visiblePtInds=unique(convhulln(P));