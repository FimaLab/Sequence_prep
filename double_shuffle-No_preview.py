import streamlit as st
import os
import pandas as pd
import random
from io import BytesIO
import matplotlib.pyplot as plt

def start(df):
    st.cache_data.clear()
    st.session_state.start = 'done'

def samples_xl_into_df(filepath, samples_col='Sample', group_col='Group'):           
    dfy = pd.read_excel(filepath, header=0, index_col=samples_col)
    samples = dfy.index.tolist()
    dfy = dfy[group_col]
    dfy.index.name = 'Sample'
    dfy.columns = ['Group']    
    if isinstance(dfy, pd.Series):
        return pd.DataFrame(data=dfy.values, index=dfy.index, columns=['Group'])
    return dfy

@st.cache_data
def randomize_samples_first(df):
    #randomize index order(sample) 
    randy = df.index.tolist()
    random.shuffle(randy)
    df_r1 = df.loc[randy]    
    return df_r1
    
#def randomize_samples_second(df):
    

def split_into_chunks(listy, n=81):
    for i in range(0, len(listy), n):
        yield listy[i:i + n]

def into_xl2(df, df2, sheet1=1, sheet2=2):
    #создает из двух таблиц кэш для записи в эксель файл
    sheet1 = str(sheet1)    
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name=sheet1, header=True, index=False)
    workbook = writer.book
    worksheet = writer.sheets[sheet1]
    df2.to_excel(writer, sheet_name=sheet2, header=True)
    writer.close()
    processed_data = output.getvalue()
    return processed_data

def ideal_in2real(real_samples_n_group, df_ideal_f='Ideal sequence.xlsx', lock_pos_to_sample=False):
    #
    df_real=pd.read_excel(df_ideal_f, header=0, index_col=False)
    sample_mask = df_real[['Sample name']]     
    finish = sample_mask.index.tolist()[-8:]        
    ideal_iter = (i for i in sample_mask.index.tolist())    
    for i in real_samples_n_group.index.tolist():            
        def nexty():
            return next(ideal_iter)            
        idy = nexty()
        while df_real.loc[idy, 'Sample name'] != 'sample':            
            idy = nexty()        
        df_real.loc[idy, 'Sample name'] = i
        df_real.loc[idy, 'Data File'] = i        
        
        if lock_pos_to_sample:
            df_real.loc[idy, 'Sample Position'] = real_samples_n_group.loc[i, 'Sample Position']
            df_real.loc[idy, 'Sample Group'] = real_samples_n_group.loc[i, 'Sample Group']
        else:
            df_real.loc[idy, 'Sample Group'] = real_samples_n_group.loc[i, 'Group']
    if len(real_samples_n_group.index.tolist()) < 81:           
        index_to_print = [i for i in range(idy+1)] + finish
        return df_real.iloc[index_to_print]
    else:
        return df_real

#@st.cache_data
def fill_sp_table(_samples_list, _frozen_table_file=False):
    def iterator(_df):
        try:
            for col in _df.columns:
                for i in _df.index:
                    yield (i, col)
        except StopIteration:
            pass
    
    _frozen_table = pd.read_excel('frozen_samples.xlsx', header=0, index_col=0)
    filled_sp_table = _frozen_table.copy()    
    t = iterator(_frozen_table)
    for sample in _samples_list:
        a = next(t)
        while not(pd.isna(filled_sp_table.loc[a])):            
            a = next(t)
        filled_sp_table.loc[a] = sample        
    filled_sp_table.fillna("",inplace=True)
    return filled_sp_table
    
def draw_sp_image(df):
    # Настройка графика
    fig, ax = plt.subplots(figsize=(11.7,8.3))    
    # Параметры окружностей
    cm = 1/2.54
    lengty = 29*cm # 10.63 icnh
    heighty = 20*cm #7.87 inch
    radius = heighty/(len(df.index) + 2)
    spacing = radius*2.1
    cols = len(df.columns)
    rows = len(df.index)
    
    # Итерируемся по индексам и значениям DataFrame
    for i, index in enumerate(df.index):
        for j, value in enumerate(df.loc[index]):
            # Вычисляем координаты окружности
            circle_x = j * spacing
            circle_y = -i * spacing            
            # Создаем окружность
            circle = plt.Circle((circle_x, circle_y), radius, edgecolor='black', facecolor='none')
            ax.add_artist(circle)
            
            # Добавляем текст внутри окружности
            ax.text(circle_x, circle_y, str(value), ha='center', va='center', fontsize=11)
            
    # Настройка осей
    ax.set_xlim(-spacing / 2, (cols -0.5)* spacing)
    ax.set_ylim(-((rows)-0.5) * spacing,  0.5*spacing)
    ax.set_aspect('equal', adjustable='box')
    
    # Добавление подписей к осям
    ax.set_xticks([i * spacing for i in range(cols)])
    ax.set_xticklabels([str(i) for i in df.columns])
    ax.set_yticks([-i * spacing for i in range(rows)])
    ax.set_yticklabels([i for i in df.index])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(top=True, labeltop=True, bottom=True, labelbottom=True, right=True, labelright=True)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    
    # Сохраняем график
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)  # Перемещение указателя в начало буфера
    return img
    
        

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    if not(start):
        first=[]
        batches= []
    with st.sidebar:        
        sample_xl_f = st.file_uploader("Choose a file", accept_multiple_files=False, 
                                       help="Excel table: sample column must be named 'Sample', group column must be named 'Group'")
        if not(sample_xl_f):
            st.cache_data.clear()
            for key in st.session_state.keys():
                del st.session_state[key]
            st.stop()
            
        else:                           
            sample_n_group = samples_xl_into_df(sample_xl_f)
            st.session_state.input_df = True
        with st.container(border=True):
            #choose metod container
            methoda = st.radio("Method", ["Metaboscan_FIA_Lipids.m", "Metaboscan.m"])
            st.write(f"Chosen: {methoda}")
        if 'input_df' in st.session_state:
            start = st.button('  Go!  ', on_click=start, args=[sample_n_group])
            
    if 'start' in st.session_state:            
        first = randomize_samples_first(sample_n_group)       
        batches = list((split_into_chunks(first.index.tolist())))
        st.session_state.ran = True
    
    
    c1, c2, c3 = st.columns([0.16, 0.42, 0.42])    
    with c1:
        st.write('Input dataframe')
        if sample_xl_f is not None:
            st.dataframe(sample_n_group, use_container_width=True)
    with c2:
        with st.container(border=True):
            c21, c22 = st.columns([0.4,0.6])
            with c21:
                st.write('Preview')
                if 'ran' in st.session_state:                    
                    st.write(first)                    
            with c22:
                st.write('Batches')                
                if 'ran' in st.session_state:
                    for i, batch in enumerate(batches,1):
                        st.write(f"{i}.Batch #{i}. Samples :{len(batch)}")

    with c3:        
        with st.container(border=True):
            st.write('Results')
            c31, c32 = st.columns([0.5,0.5])
            with c31:
                st.write('Preparation table')
                if 'ran' in st.session_state:
                    for i, batch in enumerate(batches,1):
                        #st.write(f"{i}.Batch #{i}. Samples :{len(batch)}")
                        #st.write(batch)
                        filly = fill_sp_table(batch)
                        imgy = draw_sp_image(filly)
                        btn = st.download_button(
                        label=f"Download batch {i} preparation",
                        data=imgy,
                        file_name=f'{i}_sample_preparation.png',
                        mime="image/png", key=f'prep_image{i}'
                        )
                    else:
                        pass
            with c32:
                st.write('Final table')
                if 'start' in st.session_state:
                    if 'ran' in st.session_state:
                        for i, batch in enumerate(batches, 1):                        
                            #batch list of samples, batch_n_group - df, real - df with numbers as index, real_n_group - df with 
                            batch_n_group = sample_n_group.loc[batch]                    
                            real = ideal_in2real(batch_n_group)
                            #st.write(real.loc[real['Sample name'].isin(batch)])                    
                            real_position_group = real.loc[real['Sample name'].isin(batch)][['Sample name','Sample Position', 'Sample Group']].set_index('Sample name')
                            
                            batch_n_group_pos_rand = randomize_samples_first(real_position_group)
                            real_rand = ideal_in2real(batch_n_group_pos_rand, lock_pos_to_sample=True)
                            xl = into_xl2(real_rand, batch_n_group, f'Final', f'Preparation')
                            st.download_button(label=f"Download batch # {i}", data=xl, file_name= f'batch_{i}.xlsx', key=f'xl{i}')



    
    


