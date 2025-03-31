import pandas as pd
import numpy as np
import streamlit as st
def calculate_metabolite_ratios(data):  
  #Arginine metabolism
  data['Arg/Orn+Cit']=data['Arginine']/(data['Ornitine']+data['Citrulline'])
  data['Arg/Orn']=data['Arginine']/data['Ornitine']
  data["Arg/ADMA"]=data['Arginine']/data['ADMA']
  data['(Arg+HomoArg)/ADMA']=(data['Arginine']+data['Homoarginine'])/data['ADMA']
  #Acylcarnitines
  data['C0/(C16+C18)']=data['C0']/(data['C16']+data['C18'])
  data['(C16+C18)/C2']=(data['C16']+data['C18'])/data['C2']
  data['(C6+C8+C10)/C2']=(data['C6']+data['C8']+data['C10'])/data['C2']
  data['C3/C2']=data['C3']/data['C2']
  #Tryptophan metabolism
  data['Trp/Kyn']=data['Tryptophan']/data['Kynurenine']
  data['Trp/(Kyn+QA)']=data['Tryptophan']/(data['Kynurenine']+data['Quinolinic acid'])
  data['Kyn/Quin']=data['Kynurenine']/data['Quinolinic acid']
  data['Quin/HIAA']=data['Quinolinic acid']/data['HIAA']
  #Amino acids
  data['Aspartate/Asparagine']=data['Aspartic acid']/data['Asparagine']
  data['Glutamine/Glutamate']=data['Glutamine']/data['Glutamic acid']
  data['Glycine/Serine']=data['Glycine']/data['Serine']
  data['GSG_index']=data['Glutamic acid']/(data['Serine']+data['Glycine'])

  data['Phe/Tyr']=data['Phenylalanine']/data['Tyrosin']
  data['Phe+Tyr']=data['Phenylalanine']+data['Tyrosin']
  data['BCAA']=data['Summ Leu-Ile']+data['Valine']
  data['BCAA/AAA']=(data['Valine']+data['Summ Leu-Ile'])/(data['Phenylalanine']+data['Tyrosin'])
  #Betaine_choline metabolism
  data['Betaine/choline']=data['Betaine']/data['Choline']
  return data

# Streamlit app
st.title("Метаболитные соотношения")

# Загрузка файла Excel
uploaded_file = st.file_uploader("Выберите файл Excel", type=["xlsx"])

if uploaded_file is not None:
    # Чтение данных из Excel
    data = pd.read_excel(uploaded_file)

    # Вывод исходного DataFrame
    st.subheader("Исходные данные")
    st.dataframe(data)

    # Расчет метаболитных соотношений
    output_data = calculate_metabolite_ratios(data)

    # Вывод обновленного DataFrame
    st.subheader("Обновленные данные с метаболитными соотношениями")
    st.dataframe(output_data)

# Запуск приложения
if __name__ == "__main__":
    st.write("Загрузите файл")