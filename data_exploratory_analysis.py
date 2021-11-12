#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 11:59:38 2021

@author: mohamednacereddinetoros
"""

import numpy as np
import pandas as pd
import matplotlib as plt
import statsmodels.robust as rb
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as sts
import math
from sklearn.decomposition import PCA
import plotnine as p9
import seaborn as sns
from statistics import mode
import random


donneesCabneSucre = pd.read_csv('CabaneASucrev0r2.csv')


statDonneesCabneSucre = donneesCabneSucre.describe()
dimensions = donneesCabneSucre.shape
nomsvariables = pd.DataFrame(donneesCabneSucre.columns)

classe_Sirop_list = ['Ambré','Foncé','Très Foncé','Doré']

donneesCabneSucre1 = donneesCabneSucre[donneesCabneSucre['Classe Sirop'].isin(classe_Sirop_list)]



"Mesure des indices de qualité pour Fournisseur 1"

"Calcul du degré de complitude pour l'ensemble du jeu données global"

NR = dimensions[0]

Nnan = donneesCabneSucre1.isnull().sum()

DegCompletude = 1 - (Nnan/NR)

donneesCabneSucreClean1 = donneesCabneSucre1.dropna()



#donneesCabneSucre3 = donneesCabneSucre1[donneesCabneSucre1.apply(lambda x: x.values() not in donneesCabneSucre2.values.tolist(), axis=1)]

#donneesCabneSucre00 = donneesCabneSucre1[donneesCabneSucre1.apply(lambda x: x.values.tolist() not in donneesCabneSucre2.values.tolist(), axis=1)]

"Calcul du degré de cohérence entre la tenperature min et max"
diff= donneesCabneSucreClean1["Temp max.(°C)"] - donneesCabneSucreClean1["Temp min.(°C)"] - donneesCabneSucreClean1["Diff Temp (°C)"]

diff = round(diff,2)

NNC=sum( i != 0 for i in diff)

DegCoherence=(NR-NNC)/NR

diff=pd.DataFrame(diff)

diff.index = donneesCabneSucreClean1.index

diff.columns = ["Différence"]

donneesCabneSucreClean1 = donneesCabneSucreClean1[(diff["Différence"] == 0)]


donneesCabneSucre2 = donneesCabneSucreClean1[donneesCabneSucre1.columns.drop(list(donneesCabneSucreClean1.filter(regex= 'Pixel')))]




"===========================Q1================================"

"Q1.1 : Débit de sève (L/j);"
"Q1.2 : Sucre dans la sève (%);"
"Q1.3 : % de transmittance du sirop (grade);"
"Q1.4 : La productivité en sève par saison (L/entaille)."


X = donneesCabneSucre2

X = X.drop("Débit sève (L/j)",1)

X = X.drop("Date",1)

X = X.drop("Année",1)

X = X.drop("Classe Sirop",1)


"======Q1.1========="
Y = donneesCabneSucre2["Débit sève (L/j)"]

                             
from statsmodels.stats.outliers_influence import variance_inflation_factor

VIF = [ variance_inflation_factor(X.values , i) for i in range(X.shape[1])]

VIFPanda = pd.DataFrame(VIF)

VIFPanda.index = X.columns

VIFPanda.columns =["VIF"]

VIFPanda = VIFPanda.sort_values('VIF',ascending=0)

VIFPanda1 = pd.DataFrame(VIFPanda[(VIFPanda["VIF"] < 10)])
"======Q1.1========="

import statsmodels.api as sm

modele = sm.OLS(Y,X.assign(const = 1))

resultats = modele.fit()

Y_chap = resultats.predict(X.assign(const = 1))

resultats.summary()

"====Ronde1 ===> Q1.1========="

new_features = ['Jour Calendrier Saison','Temp max.(°C)','Temp min.(°C)','Précip. Tot. Hiver (mm)','Précip. tot. (mm)','Production moyenne par entaille (L)','Sucre sève (%)','Temps bouilloire (h)','Quantité de sirop obtenue (L)','Transmittance produit (%)']

X = pd.DataFrame(X, columns = new_features)


import statsmodels.api as sm

modele = sm.OLS(Y,X.assign(const = 1))

resultats = modele.fit()

Y_chap = resultats.predict(X.assign(const = 1))

resultats.summary()


"====Ronde2 ===> Q1.1========="

new_features = ['Temp max.(°C)','Temp min.(°C)','Précip. Tot. Hiver (mm)','Précip. tot. (mm)','Production moyenne par entaille (L)','Sucre sève (%)','Temps bouilloire (h)','Quantité de sirop obtenue (L)']


X = pd.DataFrame(X, columns=new_features)

import statsmodels.api as sm

modele = sm.OLS(Y,X.assign(const = 1))

resultats = modele.fit()

Y_chap = resultats.predict(X.assign(const = 1))

resultats.summary()


"====Ronde3 ===> Q1.1========="
new_features = ['Temp max.(°C)','Temp min.(°C)','Précip. tot. (mm)','Sucre sève (%)','Temps bouilloire (h)','Quantité de sirop obtenue (L)']

X = pd.DataFrame(X, columns=new_features)

import statsmodels.api as sm

modele = sm.OLS(Y,X.assign(const = 1))

resultats = modele.fit()

Y_chap = resultats.predict(X.assign(const = 1))

resultats.summary()


"====Ronde4 ===> Q1.1========="
new_features = ['Temp max.(°C)','Temp min.(°C)','Précip. tot. (mm)']

X = pd.DataFrame(X, columns=new_features)

import statsmodels.api as sm

modele = sm.OLS(Y,X.assign(const = 1))

resultats = modele.fit()

Y_chap = resultats.predict(X.assign(const = 1))

resultats.summary()





"======= fin Validation apres ronde 2 par calcul a nouveau les valeurs des VIF  ======="
Y = donneesCabneSucre2["Débit sève (L/j)"]
                             
from statsmodels.stats.outliers_influence import variance_inflation_factor

VIF = [ variance_inflation_factor(X.values , i) for i in range(X.shape[1])]

VIFPanda = pd.DataFrame(VIF)

VIFPanda.index = X.columns

VIFPanda.columns =["VIF"]

VIFPanda = VIFPanda.sort_values('VIF',ascending=0)
VIFPanda1 = pd.DataFrame(VIFPanda[(VIFPanda["VIF"] < 10)])


"====Ronde3 ===> Q1.1========="

new_features = ['Temp max.(°C)','Temp min.(°C)','Précip. tot. (mm)']

X = pd.DataFrame(X, columns=new_features)

import statsmodels.api as sm

modele = sm.OLS(Y,X.assign(const = 1))

resultats = modele.fit()

Y_chap = resultats.predict(X.assign(const = 1))

resultats.summary()

"======= Validation apres ronde 3 par calcul a nouveau les valeurs des VIF ======="

Y = donneesCabneSucre2["Débit sève (L/j)"]
                             
from statsmodels.stats.outliers_influence import variance_inflation_factor

VIF = [ variance_inflation_factor(X.values , i) for i in range(X.shape[1])]

VIFPanda = pd.DataFrame(VIF)

VIFPanda.index = X.columns

VIFPanda.columns =["VIF"]

VIFPanda = VIFPanda.sort_values('VIF',ascending=0)
VIFPanda1 = pd.DataFrame(VIFPanda[(VIFPanda["VIF"] < 10)])

"======Début Q1.2========="

"====== Ronde 0 ===> Q1.2 ========="

X = donneesCabneSucre2

X = X.drop("Sucre sève (%)",1)

X = X.drop("Date",1)

X = X.drop("Année",1)

X = X.drop("Classe Sirop",1)

import statsmodels.api as sm

Y = donneesCabneSucre2["Sucre sève (%)"]

modele = sm.OLS(Y,X.assign(const = 1))

resultats = modele.fit()

Y_chap = resultats.predict(X.assign(const = 1))

resultats.summary()

from statsmodels.stats.outliers_influence import variance_inflation_factor

VIF = [ variance_inflation_factor(X.values , i) for i in range(X.shape[1])]

VIFPanda = pd.DataFrame(VIF)

VIFPanda.index = X.columns

VIFPanda.columns =["VIF"]


VIFPanda = VIFPanda.sort_values('VIF',ascending=0)
VIFPanda1 = pd.DataFrame(VIFPanda[(VIFPanda["VIF"] < 10)])

"====Ronde1===> Q1.2========="
new_features = ['Précip. tot. (mm)','Précip. Tot. Hiver (mm)','Nombre épisodes gel/dégel','Débit sève (L/j)','Quantité de sirop obtenue (L)']

X = pd.DataFrame(X, columns=new_features)

import statsmodels.api as sm

modele = sm.OLS(Y,X.assign(const = 1))

resultats = modele.fit()

Y_chap = resultats.predict(X.assign(const = 1))

resultats.summary()

"======= Validation apres ronde 1 par calcul a nouveau les valeurs des VIF  ======="
Y = donneesCabneSucre2["Sucre sève (%)"]
                             
from statsmodels.stats.outliers_influence import variance_inflation_factor

VIF = [ variance_inflation_factor(X.values , i) for i in range(X.shape[1])]

VIFPanda = pd.DataFrame(VIF)

VIFPanda.index = X.columns

VIFPanda.columns =["VIF"]

VIFPanda = VIFPanda.sort_values('VIF',ascending=0)
VIFPanda1 = pd.DataFrame(VIFPanda[(VIFPanda["VIF"] < 10)])

"====Ronde2 ===> Q1.2========="
new_features = ['Précip. tot. (mm)','Précip. Tot. Hiver (mm)','Nombre épisodes gel/dégel']

X = pd.DataFrame(X, columns=new_features)

import statsmodels.api as sm

modele = sm.OLS(Y,X.assign(const = 1))

resultats = modele.fit()

Y_chap = resultats.predict(X.assign(const = 1))

resultats.summary()

"======= Validation apres ronde 2 par calcul a nouveau les valeurs des VIF  ======="
Y = donneesCabneSucre2["Sucre sève (%)"]
                             
from statsmodels.stats.outliers_influence import variance_inflation_factor

VIF = [ variance_inflation_factor(X.values , i) for i in range(X.shape[1])]

VIFPanda = pd.DataFrame(VIF)

VIFPanda.index = X.columns

VIFPanda.columns =["VIF"]

VIFPanda = VIFPanda.sort_values('VIF',ascending=0)
VIFPanda1 = pd.DataFrame(VIFPanda[(VIFPanda["VIF"] < 10)])


"======= Validation apres ronde 3 par calcul a nouveau les valeurs des VIF  ======="
Y = donneesCabneSucre2["Sucre sève (%)"]
                             
from statsmodels.stats.outliers_influence import variance_inflation_factor

VIF = [ variance_inflation_factor(X.values , i) for i in range(X.shape[1])]

VIFPanda = pd.DataFrame(VIF)

VIFPanda.index = X.columns

VIFPanda.columns =["VIF"]

VIFPanda = VIFPanda.sort_values('VIF',ascending=0)
VIFPanda1 = pd.DataFrame(VIFPanda[(VIFPanda["VIF"] < 10)])



"======fin ==> Q1.2========="

"====== Ronde 0 ===> Q1.3========="

X = donneesCabneSucre2

X = X.drop("Transmittance produit (%)",1)

X = X.drop("Date",1)

X = X.drop("Année",1)

X = X.drop("Classe Sirop",1)


Y = donneesCabneSucre2["Transmittance produit (%)"]
                             
from statsmodels.stats.outliers_influence import variance_inflation_factor

VIF = [ variance_inflation_factor(X.values , i) for i in range(X.shape[1])]

VIFPanda = pd.DataFrame(VIF)

VIFPanda.index = X.columns

VIFPanda.columns =["VIF"]

VIFPanda = VIFPanda.sort_values('VIF',ascending=0)
VIFPanda1 = pd.DataFrame(VIFPanda[(VIFPanda["VIF"] < 10)])


import statsmodels.api as sm

Y = donneesCabneSucre2["Transmittance produit (%)"]

modele = sm.OLS(Y,X.assign(const = 1))

resultats = modele.fit()

Y_chap = resultats.predict(X.assign(const = 1))

resultats.summary()

"====Ronde1===> Q1.3========="

new_features = ['Temp max.(°C)','Temp min.(°C)','Quantité de sirop obtenue (L)']


X = pd.DataFrame(X, columns=new_features)

import statsmodels.api as sm

modele = sm.OLS(Y,X.assign(const = 1))

resultats = modele.fit()

Y_chap = resultats.predict(X.assign(const = 1))

resultats.summary()

"======= Validation apres ronde 1 par calcul a nouveau les valeurs des VIF  ======="
Y = donneesCabneSucre2["Transmittance produit (%)"]                             
from statsmodels.stats.outliers_influence import variance_inflation_factor

VIF = [ variance_inflation_factor(X.values , i) for i in range(X.shape[1])]

VIFPanda = pd.DataFrame(VIF)

VIFPanda.index = X.columns

VIFPanda.columns =["VIF"]

VIFPanda = VIFPanda.sort_values('VIF',ascending=0)
VIFPanda1 = pd.DataFrame(VIFPanda[(VIFPanda["VIF"] < 10)])


"======  Fin Q1.3========="

"====== Ronde 0 ===> Q1.4========="

X = donneesCabneSucre2

X = X.drop("Production moyenne par entaille (L)",1)

X = X.drop("Date",1)

X = X.drop("Année",1)

X = X.drop("Classe Sirop",1)


Y = donneesCabneSucre2["Production moyenne par entaille (L)"]
                             
from statsmodels.stats.outliers_influence import variance_inflation_factor

VIF = [ variance_inflation_factor(X.values , i) for i in range(X.shape[1])]

VIFPanda = pd.DataFrame(VIF)

VIFPanda.index = X.columns

VIFPanda.columns =["VIF"]

VIFPanda = VIFPanda.sort_values('VIF',ascending=0)
VIFPanda1 = pd.DataFrame(VIFPanda[(VIFPanda["VIF"] < 10)])


import statsmodels.api as sm


modele = sm.OLS(Y,X.assign(const = 1))

resultats = modele.fit()

Y_chap = resultats.predict(X.assign(const = 1))

resultats.summary()



"====Ronde2 ===> Q1.2========="
new_features = ['Précip. Tot. Hiver (mm)','Débit sève (L/j)','Nombre épisodes gel/dégel','Quantité de sirop obtenue (L)']

X = pd.DataFrame(X, columns=new_features)

import statsmodels.api as sm

modele = sm.OLS(Y,X.assign(const = 1))

resultats = modele.fit()

Y_chap = resultats.predict(X.assign(const = 1))

resultats.summary()


"====Ronde3 ===> Q1.2========="
new_features = ['Précip. Tot. Hiver (mm)','Nombre épisodes gel/dégel']

X = pd.DataFrame(X, columns=new_features)

import statsmodels.api as sm

modele = sm.OLS(Y,X.assign(const = 1))

resultats = modele.fit()

Y_chap = resultats.predict(X.assign(const = 1))

resultats.summary()


"==== Q2 ======"

"Variations de la transmittance et du contenu en sucre de la sève d’une année à l’autre(2014,2015,2016)"


"============= Q2.1 ==============="
"tests d’hypothèse pour le sucre de la sève "

X2 = donneesCabneSucre2


Q1 = X2["Sucre sève (%)"].quantile(0.25)
Q3 = X2["Sucre sève (%)"].quantile(0.75)
IQR = Q3 - Q1

X2 = X2[(X2["Sucre sève (%)"] > (Q1 - 1.5 * IQR)) & (X2["Sucre sève (%)"] < (Q3 + 1.5 * IQR))]

X2 = X2[(X2["Année"]>= 2014) & (X2["Année"]<= 2016)]

ax = X2.boxplot(by="Année", column="Sucre sève (%)")
ax.set_xlabel('Année')
ax.set_ylabel("Sucre sève (%)")
plt.title('')

sucre2014 = X2[(X2["Année"]==2014)]["Sucre sève (%)"].values.reshape(-1,1)


sucre2015 = X2[(X2["Année"]==2015)]["Sucre sève (%)"].values.reshape(-1,1)

sucre2016 = X2[(X2["Année"]==2016)]["Sucre sève (%)"].values.reshape(-1,1)


print(sucre2014.mean())
print(sucre2015.mean())
print(sucre2016.mean())

import scipy.stats as sts

pvalue1=sts.f_oneway(sucre2014,sucre2015,sucre2016)

"============= Q2.1 ==============="
"tests d’hypothèse pour transmittance du sirop "

X02 = donneesCabneSucre2


Q1 = X02["Transmittance produit (%)"].quantile(0.25)
Q3 = X02["Transmittance produit (%)"].quantile(0.75)
IQR = Q3 - Q1

X02 = X02[(X02["Transmittance produit (%)"] > (Q1 - 1.5 * IQR)) & (X02["Transmittance produit (%)"] < (Q3 + 1.5 * IQR))]

X02 = X02[(X02["Année"]>= 2014) & (X02["Année"]<= 2016)]

ax = X02.boxplot(by="Année", column="Transmittance produit (%)")
ax.set_xlabel('Année')
ax.set_ylabel("Transmittance produit (%)")
plt.title('')

Transmittance2014 = X02[(X02["Année"]==2014)]["Transmittance produit (%)"].values.reshape(-1,1)

Transmittance2015 = X02[(X02["Année"]==2015)]["Transmittance produit (%)"].values.reshape(-1,1)

Transmittance2016 = X02[(X02["Année"]==2016)]["Transmittance produit (%)"].values.reshape(-1,1)



print(Transmittance2014.mean())
print(Transmittance2015.mean())
print(Transmittance2016.mean())

import scipy.stats as sts

pvalue01=sts.f_oneway(Transmittance2014,Transmittance2015,Transmittance2016)

"=============  Q2.2 ==============="
pvalue3=sts.ttest_ind(Transmittance2015,Transmittance2016)
pvalue4=sts.ttest_ind(Transmittance2014,Transmittance2016)

"============= Q3 ==============="


from sklearn.decomposition import PCA


X = donneesCabneSucre1[donneesCabneSucre1.columns.drop(list(donneesCabneSucre1.filter(regex= '^((?!Pixel).)*$')))]



X=pd.get_dummies(X)

MatriceR = X.corr()


Y =  donneesCabneSucre1['Transmittance produit (%)']

import statsmodels.api as sm

modele = sm.OLS(Y,X.assign(const = 1))

resultats = modele.fit()

Y_chap = resultats.predict(X.assign(const = 1))

resultats.summary()

"==============="
# from statsmodels.stats.outliers_influence import variance_inflation_factor

# VIF = [ variance_inflation_factor(X.values , i) for i in range(X.shape[1])]

# VIFPanda = pd.DataFrame(VIF)

# VIFPanda.index = X.columns

# VIFPanda.columns =["VIF"]

# VIFPanda = VIFPanda.sort_values('VIF',ascending=0)
# VIFPanda1 = pd.DataFrame(VIFPanda[(VIFPanda["VIF"] < 10)])



"==============="

Yclass = donneesCabneSucre1['Classe Sirop']

a=2

pca = PCA(n_components = a)

pca.fit(X)

VarianceCapturee = pca.explained_variance_ratio_

X_pca_mean = pca.mean_

T = pca.transform(X)

P = pca.components_


T1 = np.dot((X.values - X_pca_mean),P[0,:])

T2 = np.dot((X.values - X_pca_mean),P[1,:])

Xest = pca.inverse_transform(T)






"La première composant combine les différentes longueurs corrélées ensembles (taille du poisson), et la deuxième va chercher d’autres informations (à analyser)"

TPanda = pd.DataFrame(T)

TPanda.index = X.index

TPanda.columns =['T1','T2']

T_resultats = pd.concat([TPanda,Yclass ],axis=1)


"======PCR========"


modele=sm.OLS(Y,TPanda.assign(const=1))
resultats=modele.fit()
Y_chap = resultats.predict(TPanda.assign(const=1))
resultats.summary()



"================="



PPanda = pd.DataFrame(P.T)

PPanda.index = X.columns

PPanda.columns =['PC1','PC2']

PPanda.plot.bar()


"Première tendance (PC1): taille physique du poisson (plus les variables de dimension sont grandes, plus T1 augmente)"
"Deuxième tendance (PC2): basée sur les différences morphologiques des espèces de poisson (ex: la hauteur des poissons, entre les brochets et les daurades)"
import plotnine as p9

graph = p9.ggplot(data= T_resultats, mapping=p9.aes(x='T1', y='T2', color ='Classe Sirop'))

print (graph + p9.geom_point())















