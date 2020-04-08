data = pd.read_table("/home/coder/PenSynthPy/MLAB_data.txt")
data.head()
d = data['Treated']
X = data[["Income","RetailPrice", "Young", "BeerCons"
                  , "SmokingCons1970", "SmokingCons1971", "SmokingCons1972", "SmokingCons1973", "SmokingCons1974", "SmokingCons1975"
                  , "SmokingCons1980", "SmokingCons1988"]]

names = []
for i in range(1970,1981):
    names.append('SmokingCons' +str(i))
Z = data[names]

for i in range(1981,2001):
    names.append('SmokingCons' +str(i))
y = data[names]

V=np.diag([.1,.1,.1,.1,6,6,6,6,6,6,6,6])
sol = pensynth_weights(np.array(X[d==0].T), np.array(X[d==1].T), V, pen = .1)