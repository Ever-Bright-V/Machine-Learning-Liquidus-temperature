function [y1 y2]=UseBpModel(x,XNom_Sys,BpNet,ReverseInfo)
    Nom_X=mapminmax('apply',x,XNom_Sys) ;
	y1=sim(BpNet,Nom_X);
    y2=mapminmax('reverse',y1,ReverseInfo)
end