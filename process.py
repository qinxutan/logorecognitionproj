import csv
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools

def get_ground_truth():
    image_list = os.listdir('logos')
    total_images = len(image_list)
    df = pd.DataFrame(image_list, columns = ['imageName'])
    df['category'] = df['imageName'].str.split('-').str[0]
    df_categorised = df.groupby(['category']).count().reset_index()
    df_categorised = df_categorised.rename(columns={'imageName':'totalImages'})
    df_categorised['TrueMatch'] = 0.5*(df_categorised['totalImages']*(df_categorised['totalImages']-1)).astype(int)
    df_categorised['FalseMatch'] = df_categorised['totalImages']*(total_images-df_categorised['totalImages'])
    df_categorised = df_categorised[['category','totalImages','TrueMatch','FalseMatch']]
    df_numtrue = dict(zip(df_categorised.category, df_categorised.TrueMatch))
    df_numfalse = dict(zip(df_categorised.category, df_categorised.FalseMatch))
    return df_categorised, df_numtrue, df_numfalse

def process_df2(df2):
    df2_items = list(df2.columns)
    df2_items_dict = {}
    for i in range(1,len(df2_items)):
        df2_items_dict["Item"+str(i)] = df2_items[i]
    a = df2.iloc[:,1:].to_numpy()
    #upper triangle part includes the diagonal
    idx = np.triu_indices(a.shape[0], k = 0)
    #repeat range and filter by indices
    c = np.repeat(np.arange(1, a.shape[0]+1), a.shape[0]).reshape(a.shape)[idx]
    i = np.tile(np.arange(1, a.shape[0]+1), a.shape[0]).reshape(a.shape)[idx]
    #filter array by indices
    v = a[idx]    
    #create DataFrame
    df2 = pd.DataFrame({'seed':c, 'match': i, 'sim': v})
    #add substring
    df2[['seed','match']] = 'Item'+df2[['seed','match']].astype(str)
    df2['seed'] = df2['seed'].map(df2_items_dict)
    df2['match'] = df2['match'].map(df2_items_dict)
    df2['seed_category'] = df2['seed'].str.split('-').str[0]
    df2['match_category'] = df2['match'].str.split('-').str[0]
    df2 = df2[df2['seed']!=df2['match']]
    df2['equal'] = df2['seed_category'] == df2['match_category']
    return df2

def process_df2_results(df2,match_score):
    df2 = df2[df2['sim']>=match_score]
    df2_results = pd.DataFrame(columns=['category','TPR %','FPR %','TP', 'FP'])
    df2_results = df2_results.set_index(['category'])
    for group, sub_df in df2.groupby(['seed_category']):
        correct_matches = int(sub_df[sub_df.equal==True].shape[0]/2)
        incorrect_matches = sub_df[sub_df.equal==False].shape[0]
        try:
            tp_rate = round(correct_matches/df_numtrue[group] * 100,2)
        except:
            tp_rate = 0
        try:
            fp_rate = round(incorrect_matches/df_numfalse[group] * 100,2)
        except:
            fp_rate = 100

        df2_results.loc[group] = [tp_rate,fp_rate,correct_matches,incorrect_matches]
    return df2_results.reset_index()

def plot_overall(df1_results,df2_results_dict):
    output_file = os.path.join('results',"0_overall.png")
    title = 'overall'
    df2_overall_joined = pd.DataFrame(columns=['threshold','TPR %','FPR %']).set_index('threshold')
    for k,v in df2_results_dict.items():
        TPR = (v.TP.sum()/sum(df_numtrue.values()))*100
        FPR = (v.FP.sum()/sum(df_numfalse.values()))*100
        df2_overall_joined.loc[k] = [TPR,FPR]

    plt.plot(df1_results.FP.sum()/sum(df_numfalse.values())+CONST,(df1_results.TP.sum()/sum(df_numtrue.values()))+CONST, marker='o',color='red', markersize=20)
    
    plt.plot(df2_overall_joined["FPR %"]/100+CONST, df2_overall_joined["TPR %"]/100+CONST, marker='o',color='blue')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim([CONST,1])
    plt.ylim([0,1])
    plt.xscale("log")
    plt.grid(True, which="both")
    plt.title(title)
    plt.savefig(output_file)
    plt.clf()
    #exit()

def consolidate_results(df1_results,df2_results_dict):
    for row,sub_df in df_groundtruth.iterrows():
        category = sub_df.category
        category_postprocess = sub_df.category.split(".")[0]
        output_file = os.path.join('results',category_postprocess+".png")
        title = '{}, images={}'.format(category_postprocess,sub_df.totalImages)
        df2_category_joined = pd.DataFrame([])
        for k,v in df2_results_dict.items():
            df2_i = v[v['category']==category].copy()
            df2_i['i'] = k
            df2_category_joined = pd.concat([df2_category_joined,df2_i])
        df1_category = df1_results[df1_results['category']==category]
        plt.plot(df1_category["FPR %"]/100+CONST, df1_category["TPR %"]/100+CONST, marker='o',color='red', markersize=20)
        
        plt.plot(df2_category_joined["FPR %"]/100+CONST, df2_category_joined["TPR %"]/100+CONST, marker='o',color='blue')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.xlim([CONST,1])
        plt.ylim([0,1])
        plt.xscale("log")
        plt.grid(True, which="both")
        plt.title(title)
        plt.savefig(output_file)
        plt.clf()
    plot_overall(df1_results,df2_results_dict)


        
    #print(df_groundtruth)


if __name__ == '__main__':
    global CONST
    CONST = 1e-7

    df1 = pd.read_csv('custom.csv')
    df2 = pd.read_csv('phishintention.csv')
    global df_groundtruth, df_numtrue, df_numfalse
    df_groundtruth, df_numtrue, df_numfalse = get_ground_truth()
    print(df_groundtruth)

    df1 = df1[['seed','match']]
    df1['seed_category'] = df1['seed'].str.split('-').str[0]
    df1['match_category'] = df1['match'].str.split('-').str[0]
    df1['equal'] = df1['seed_category'] == df1['match_category']
    df1_results = pd.DataFrame(columns=['category','TPR %','FPR %', 'TP', 'FP'])
    df1_results = df1_results.set_index(['category'])
    for group, sub_df in df1.groupby(['seed_category']):
        #print(group)
        #print(sub_df.equal.value_counts())
        correct_matches = int(sub_df[sub_df.equal==True].shape[0]/2)
        incorrect_matches = sub_df[sub_df.equal==False].shape[0]
        try:
            tp_rate = round(correct_matches/df_numtrue[group] * 100,2)
        except:
            tp_rate = 0
        try:
            fp_rate = round(incorrect_matches/df_numfalse[group] * 100,2)
        except:
            fp_rate = 100
        df1_results.loc[group] = [tp_rate,fp_rate,correct_matches,incorrect_matches]
        #print(group,tp_rate, fp_rate)
    df1_results = df1_results.reset_index()
    #print(df1_results)

    # plt.matshow(df2.corr())
    # #plt.xticks(range(df2.select_dtypes(['number']).shape[1]), df2.select_dtypes(['number']).columns, fontsize=14, rotation=45)
    # #plt.yticks(range(df2.select_dtypes(['number']).shape[1]), df2.select_dtypes(['number']).columns, fontsize=14)
    # cb = plt.colorbar()
    # cb.ax.tick_params(labelsize=14)
    # plt.title('Correlation Matrix', fontsize=16)
    #plt.show()
    #plt.clf()

    df2_results_dict = {}
    df2 = process_df2(df2)
    for i in np.arange(0.01,0.99,0.01):
        i = round(i,2)
        df2_results_dict[i] = process_df2_results(df2,i)
    #print(df2_results_dict)

    consolidate_results(df1_results,df2_results_dict)