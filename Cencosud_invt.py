# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 16:39:33 2021

@author: CARLOS.MARTINEZ
"""

import pandas as pd
import re

year = '2021'
week = 'W13'
ruta_1 = 'C:\\Users\\carlos.martinez\\Documents\\MASTER DATA\\{}'
ruta_2 = 'D:\\BI_DATA\\Cencosud\\weekly\\AV\\{}'
ruta_3 = 'D:\\BI_DATA\\Cencosud\\weekly\\STOCK\\AV\\{}'

DataPOS=pd.ExcelFile(ruta_1.format('BI-MASTER POS.xlsx'))
DataDiccionario=pd.ExcelFile(ruta_1.format('Material CE.xlsx'))
DataCencosud = pd.ExcelFile(ruta_2.format('Cencosud_AV_'+year+'_'+week+'.xlsx'))
DataEOL=pd.ExcelFile(ruta_1.format('W45 EOL consolidado.xlsx'))
DataPuntos=DataPOS.parse('POS')
DataPuntosCencosud = DataPOS.parse('CENCOSUD DEPENDENCIAS')
DataEAN=DataDiccionario.parse('EAN MATERIAL MASTER')
DataDate=DataDiccionario.parse('DATES')
DataInventario = DataCencosud.parse('Inventario')
DataEOLAV=DataEOL.parse('AV')
Datadep=DataDiccionario.parse('DEP CENCOSUD')


###CONDICIONALES PARA CAMBIO DE NOMBRE DE COLUMNA, YA QUE LA INFROMACIÓN VARÍA SEGÚN LA PERSONA QUE LA ENVÍA###
###AUNQUE EN ESTE PUNTO LA INFORMACIÓN QUE SE TIENE ES MUY LIMITADA, MÁS ADELANTE LLEGARÁ EL PROCESO NATURAL CON TODOS LOS ESCENARIOS###

DataInventario.columns = DataInventario.columns.str.lower() #NUEVO!!!#

DataInventario.rename(columns={j: re.sub(r'[^\w\s]','', j) for j in DataInventario.columns.tolist()}, inplace = True)##!!!####

DataInventario.rename(columns={DataInventario.columns[1]: 'dependencia'}, inplace = True)

DataInventario.convert_dtypes().dtypes

if 'dependencia' in DataInventario:
    DataInventario['dependencia'] = DataInventario['dependencia'].astype('str')
    DataInventario['cod tienda'] = ''
    for i,x in DataInventario.iterrows():
        d = re.sub(r'[a-zA-Z_\W].*','',x['dependencia'])
        DataInventario['cod tienda'][i] = d
        
#f = re.sub(r'[a-zA-Z_]')
            
#d = re.sub(r'^\d{1,3}','a',x['nombre tienda']) 
#d = re.sub(r'[a-zA-Z_-]','',x['nombre tienda'])
#d = re.search(r'^\d{1,3}',x['nombre tienda']) 
    
#if 'codigo_cliente' in DataInventario:
#    DataInventario.rename(columns={'codigo_cliente':'Cod Tienda'}, inplace=True)
    
#if 'Ean' in DataInventario:
#    DataInventario.rename(columns={'Ean':'ean'}, inplace=True)

###CREANDO COLUMNA DAMAGED, PARA LOS CODIGOS DAÑADOS VENDIBLES, COMO TODOS LOS COD DE DAÑADO VENDIBLES TIENEN 8 DÍGITOS Y,####
###NO SUPERAN LOS 30000000, DEJO ESTA CONSTANTE, AUNQUE EN EL MOMENTO QUE LLEGUE A SUCEDER TOCARÍA ACTUALIZARLA###        
damaged=[]

for i in range(len(DataInventario)):
    if DataInventario.at[i,'ean'] < 30000000:
        damaged.append('Y')
    else:
        damaged.append('N')
        
DataInventario['state'] = damaged
    
###SE CREAN LAS CONSTANTES, UNAS COMO YEARWEEK SIRVE DE LLAVE DE CRUCE DE DATOS PARA LOS VECTORES DE TIEMPO###
###EN EL RESTO SIMPLEMENTE SE REALIZAN LOS CRUCES DE BASES DE DATOS NORMALES, CONDICIONADOS CLARO A LOS DICCIONARIOS###

DataInventario['cod tienda'] = DataInventario['cod tienda'].astype('int')
DataInventario['yearWeek'] = year+'-'+week
DataInventario['source'] = 'CUSTOMER'
    
DataInventario=DataInventario.merge(DataPuntosCencosud[['Dependencia',
                                                        'Cod POS']],
                                    how='left', left_on='cod tienda', right_on='Dependencia').drop('Dependencia',1)

DataInventario=DataInventario.merge(DataPuntos[['Code_Store',
                                                'Name_Store',
                                                'Type_1',
                                                'Account (AV)']],
                                    how='left', left_on='Cod POS', right_on='Code_Store').drop('Cod POS',1)

DataInventario=DataInventario.merge(DataEAN[['model',
                                             'brand',
                                             'productGroup',
                                             'subGroup_2',
                                             'ean',
                                             'category']],
                how='left', on='ean')

DataInventario=DataInventario.merge(DataDate[['date',
                                              'year',
                                              'week',
                                              'yearWeek',
                                              'month']],how='left',
                on='yearWeek')

######### STOCK NOT MERGE WITH SIMPLE EAN##############

DataInventario_2=DataInventario[DataInventario['model'].isnull()]

DataInventario_2.drop(['model',
                       'brand',
                       'productGroup',
                       'subGroup_2',
                       'category'], axis= 'columns', inplace=True)

DataInventario_2=DataInventario_2.merge(Datadep[['dep',
                                                 'ean']],
                                    how='left', left_on='ean', right_on='dep').drop('dep',1)

DataInventario_2.rename(columns={'ean_y':'ean'}, inplace=True)

DataInventario_2=DataInventario_2.merge(DataEAN[['model',
                                             'brand',
                                             'productGroup',
                                             'subGroup_2',
                                             'ean',
                                             'category']],
                how='left', on='ean')

DataTrash=DataInventario_2[DataInventario_2['model'].isnull()]
DataTrash_2= DataInventario[DataInventario['Code_Store'].isnull()]

DataTrash = pd.DataFrame(DataTrash, columns=('ean_x','articulo'))
DataTrash_2 = pd.DataFrame(DataTrash_2, columns=('cod tienda','dependencia','tienda',))

##SE ELIMINANA DUPLICADOS PARA VER MÁS FÁCIL LOS DATOS SOBRANTES###

DataTrash =DataTrash.drop_duplicates(['ean_x', 'articulo'])


###AL HACER PRUEBAS OBLIGATORIAMENTE ME TOCÓ DEJAR ESTE PASÓ, LO SÉ, ES UNA MALA PRÁCTICA REPETIR COD, PERO ERA NECESARIO###
###YA QUE DE ESTA MANERA SE LEEN TODOS LOS ARCHIVOS SIN IMPORTAR QUE LAS OTRAS COLUMNAS ORIGINALES CONTENGAN NA###

DataInventario.rename(columns={'Code_Store':'CodePos','Name_Store':'pos',
                              'Account (AV)':'account','Type_1':'place',
                              'stock unidades':'Unidades'}, inplace=True)

DataInventario_2.rename(columns={'Code_Store':'CodePos','Name_Store':'pos',
                              'Account (AV)':'account','Type_1':'place',
                              'stock unidades':'Unidades'}, inplace=True)

DataInventario=pd.DataFrame(DataInventario, columns=('category','source','date','year','month',
                                       'week','yearWeek','CodePos','pos',
                                       'account','place','ean','model',
                                       'brand','productGroup','subGroup_2','Unidades','state'))

DataInventario_2=pd.DataFrame(DataInventario_2, columns=('category','source','date','year','month',
                                       'week','yearWeek','CodePos','pos',
                                       'account','place','ean','model',
                                       'brand','productGroup','subGroup_2','Unidades','state'))

####SE CONCATENAN LOS DOS DF Y SE FORMA UN ARCHIVO COMPLETO, TANTO LOS EAN MAPEADOS, COMO LOS DEV CENCOSUD####
###LO BUENO DE ESTO ES QUE AL CONCATENAR E IGNORAR EL INDEX, SE RECREA UNO COMPLETAMENTE CONSECUTIV###

todos = [DataInventario, DataInventario_2]
datatotal = pd.concat(todos,ignore_index='True')

###AL CONCATENAR LAS BAASES DE DATOS, LO BUENO DE ESTO ES QUE LOS DOS DF AL IGNORAR EL INDEX RAIZ CREAN UN SOLO INDEX, Y NO AFECTA LA ITERACIÓN DE EOL###
###SE CREA LA COLUMNA EOL Y SE RENOMBRA####

datatotal = datatotal.merge(DataEOLAV[['ean','EOL']], how='left', on='ean')
    
AEol=[]
for i in range(len(datatotal)):
    if datatotal.at[i,'EOL']=='EOL':
        AEol.append('Y')
    else:
        AEol.append('N')
        
datatotal['EOL_2'] = AEol
  
datatotal.drop(['EOL'],axis= 'columns', inplace=True)

datatotal.rename(columns={'EOL_2':'EOL'}, inplace=True)

###SE CREA UN DF CON LAS COLUMNAS REQUERIDAS POR EL FORMATO DEL LIDER BI###
###CUIDADO!! ESTOS FORMATOS CAMBIAN TODAS LAS SEMANAS, IGUAL QUE LA CONFIGURACIÓN DE LAS COLUMNAS DEL DICCIONARIO Y TODO FALLA###
### PASO IMPORTANTE, BORRAR TODOS LOS NA, CON ESTO QUITAMOS DUPLICADOS QUE SE GENERARON AL CONCATENAR LOS DOS DF###

datatotal.dropna(inplace=True)
datatotal=datatotal.drop(datatotal[datatotal['category']=='OTHER'].index)
datatotal.drop(['category'],axis= 'columns', inplace=True)

###YESSSS!!! SE IMPORTAN TODOS LOS ARCHIVOS Y ASI TERMINA EL SCRIPT, AL QUE LO USE, ESPERO LO SEPAS APRECIAR#
###TIENE TECNOLOGÍAS QUE JAMÁS VAS A ENTENDER NI PODER IMPLEMENTAR, HAY QUE ESTUDIAR ;)#####
datatotal.to_excel(ruta_3.format('Cencosud_Invent_AV'+'_'+year+'_'+week+'.xlsx'), index=False)
DataTrash.to_excel(ruta_3.format('DataTrash_AV'+'_'+year+'_'+week+'.xlsx','ean'), index=False)
#DataTrash_2.to_excel(ruta_3+'DataTrash_'+'_'+year+'_'+week+'.xlsx', 'pos')
