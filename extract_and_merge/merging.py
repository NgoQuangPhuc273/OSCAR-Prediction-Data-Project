import pandas as pd
import numpy as np

imdb = pd.read_csv('csv/imdb.csv')
oscar = pd.read_csv('csv/oscar.csv')
numbers = pd.read_csv('csv/numbers.csv')
bafta = pd.read_csv('csv/BAFTA.csv')
dga = pd.read_csv('csv/DGA.csv')
ggCom = pd.read_csv('csv/GG_comedy.csv')
ggDrm = pd.read_csv('csv/GG_drama.csv')
pga = pd.read_csv('csv/PGA.csv')
gpalm = pd.read_csv('csv/GPalm.csv')
ccma = pd.read_csv('csv/CCMA.csv')

#make each to list
oscarList = oscar['film'].tolist()
imdbList = imdb['Movie'].tolist()

nominee=[]

#append in for nominees
for movie in imdbList:
    if movie in oscarList:
        nominee.append(1)
    else:
        nominee.append(0)

#list of winners
champs = oscar[oscar['oscar_winner']==1]
oscar_winners = champs['film'].tolist()

winner = []
#append in for winners
for movie in imdbList:
    if movie in oscar_winners:
        winner.append(1)
    else:
        winner.append(0)

imdb['Oscar_winner'] = pd.DataFrame(np.array(winner))
imdb['Oscar_nominee'] = pd.DataFrame(np.array(nominee))

imdbList = imdb['Movie'].tolist()

def get_winNom(imdbList, sourceDF):
    winner = []
    nominee = []
    champs = []
    #list of all movies in the sourceDF
    sourceList = sourceDF['film'].tolist()
    #to append in 1/0 for nominees
    for movie in imdbList:
        if movie in sourceList:
            nominee.append(1)
        else:
            nominee.append(0)
            
    #create a list of only winners
    champs = sourceDF[sourceDF['winner']==1]
    champs = champs['film'].tolist()
    for movie in imdbList:
        if movie in champs:
            winner.append(1)
        else:
            winner.append(0)
            
    return winner, nominee

def remove_char(numberdf):
    number = numberdf.replace('\xa0','', regex = True)
    number = number.replace('\Â', '', regex=True)
    number = number.replace('\$', '', regex=True)
    number = number.replace('\,', '', regex=True)
    number = number.astype('int64')
    return number

def get_money(imdbList, numbersDF):
    budget_mon = []
    dom_mon = []
    world_mon = []
    #remove the unusual character first
    budDF = numbersDF['budget']
    domesDF = numbersDF['domestic_gross']
    worldDF = numbersDF['world_wide_gross']
    budDF = remove_char(budDF)
    domesDF = remove_char(domesDF)
    worldDF = remove_char(worldDF)
    
    #remake into list
    domestic = domesDF.tolist()
    worldwide = worldDF.tolist()
    budget = budDF.tolist()
    film = numbersDF['movie_name'].tolist()
    
    for movie in imdbList:
        i = 0
        while (i<5700):
            if (movie==film[i]):
                break
            i+=1
        if(i<5700):
            budget_mon.append(budget[i])
            dom_mon.append(domestic[i])
            world_mon.append(worldwide[i])
        else:
            budget_mon.append(0)
            dom_mon.append(0)
            world_mon.append(0)
    return budget_mon, dom_mon, world_mon

#get money data
myBudget, myDomestic, myWorld = get_money(imdbList, numbers)

#for bafta award
bafta_winner, bafta_nom = get_winNom(imdbList, bafta)

#for dga
dga_winner, dga_nom = get_winNom(imdbList, dga)

#for ggCom
ggCom_winner, ggCom_nom = get_winNom(imdbList, ggCom)

#for ggDrm
ggDrm_winner, ggDrm_nom = get_winNom(imdbList, ggDrm)

#for pga
pga_winner, pga_nom = get_winNom(imdbList, pga)

#for ccma
ccma_winner, ccma_nom = get_winNom(imdbList, ccma)

#golden palm
gpalm_winner, gpalm_nom = get_winNom(imdbList, gpalm)

award_list = [bafta_winner, bafta_nom, dga_winner, dga_nom, 
       ggCom_winner, ggCom_nom, ggDrm_winner, ggDrm_nom, 
       pga_winner, pga_nom, gpalm_winner, gpalm_nom,
       ccma_winner, ccma_nom, myBudget, myDomestic, myWorld]

col_Names = ["BAFTA_winner", "BAFTA_nominee", "DGA_winner", "DGA_nominee",
             "GG_comedy_winner", "GG_comedy_nominee", "GG_drama_winner", "GG_drama_nominee", 
             "PGA_winner", "PGA_nominee", "Golden_Palm_winner", "Golden_Palm_nominee",
             "CCMA_winner", "CCMA_nominee", "Budget","Domestic (US) gross", "Worldwide gross"]

mini = 0
maxi = 17
for i in range(mini, maxi):
    df = pd.DataFrame(np.array(award_list[i]), columns={""+col_Names[i]+""})
    imdb = pd.concat([imdb, df],axis=1)
    
imdb["International gross"] = " "
imdb['International gross'] = imdb['Worldwide gross'].sub(imdb['Domestic (US) gross'])

imdb["Golden_Bear_winner"] = 0
imdb["Golden_Bear_nominee"] = 0
imdb["Golden_Lion_winner"] = 0
imdb["Golden_Lion_nominee"] = 0
imdb["PCA_winner"] = 0    
imdb["PCA_nominee"] = 0  
imdb["NYFCC_winner"] = 0
imdb["NYFCC_nominee"] = 0 
imdb["OFCS_winner"] = 0    
imdb["OFCS_nominee"] = 0

imdb.to_csv('csv/final_movie_dataset.csv', index=False)