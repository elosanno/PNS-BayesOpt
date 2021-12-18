function v = lowtri2vec_inchol(L,D,irank)

v = [];
for i = 1 : D
  v = [v;L(i,1:min(i,irank))'];
end
