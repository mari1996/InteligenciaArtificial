
%Colocar no parametro a dimensao
sD = som_read_data('dados.txt',4);

%Normalização dos dados
sD = som_normalize(sD, 'var');

%Treinamento do dado
sM = som_make(sD);

%Label do SOM
sM = som_autolabel(sM, sD, 'vote');

%Visualizacao basica ( o 2 ali é o n.o de dimensao, de novo)
som_show(sM,'umat','all','comp',1:2,'empty','Labels','norm','d');
%ver como corrigir isso: som_show_add('label',sM,'subplot',6);

[Pd,V,me] = pcaproj(sD, 3);
%plotagem 1
som_grid(sM, 'Coord', pcaproj(sM,V,me),'marker','none','Label',sM.labels,'labelcolor','k');

%plotagem 2
%denormaliza os pesos dos vetores
M = som_denormalize(sM.codebook, sM);

%Faz um array dos markers baseados na subespecie
colM = zeros(length(sM.codebook), 3);
un = unique(sD.labels);

for i=1:3, ind = find(strcmp(sM.labels, un(i))); colM(ind,i) = 1; end

%plotagem
som_grid(sM, 'Coord', M(:,2:4),'MarkerSize',(M(:,1)-4)*5,'Markercolor', colM);