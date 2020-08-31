
import logging
import numpy
import calendar, time
import math
import pandas as pd

schemas = {}
schem_def = {}
dataframes = {}


import logging
logging.basicConfig(filename='oreilly.log',level=logging.INFO)

def report_context(current_runtime,process_name,project_name,environment):

    logging.info("------Execution context:------")
    logging.info("Process Running: %s in environment %s" %(process_name,environment))
    
    global process_pk
    process_pk = process_name

    logging.info("Project: %s" %project_name)
    projectpk = project_name

def report_data_source(df,df_name):

    logging.info("------New reported data source:------")
    
    dsl = df_name.split('.')

    name=dsl[0]
    format_ds=dsl[1]

    ds_pk = name
    
    fields = df.dtypes.to_dict()
    
    schema_field = [(el,str(fields[el])) for el in fields]

    schema_pk = name
    schemas[dsl[0]]=schema_pk
    schem_def[dsl[0]]=df.columns.to_list()
    dataframes[dsl[0]]=df

    logging.info("DataSource %s, its format is %s and its schema is %s" %(name,format_ds,schema_field))


def report_link(froms,to, current_runtime,stats_for_to=True):
    from_schemas = [schem_def[frm] for frm in froms]
    to_schema = schem_def[to]

    logging.info("------New link between data sources:------")
    
    
    name_lin = 'From '+str(froms)+' to '+to
    
    from_pks = [schemas[frm] for frm in froms]
    to_pk = schemas[to]    

    logging.info("New lineage: %s" %name_lin)
    
    stats_for = froms

    if stats_for_to:

        stats_for = froms +[to]
        
    

    for elem in stats_for:
        df = dataframes[elem]
        #print(df)
        statistics = {}
        e=df.describe().to_dict()
        for el in e:
            for st in e[el]:

                if not math.isnan(e[el][st]):
                    statistics[el+'.'+st] = e[el][st]

        logging.info("New statistics linked to the execution: %s which refers to schema %s" %(statistics,elem))             


def report_model(train,test,x_test,y_test,model,model_path,current_runtime):
    
    import pandas as pd
    logging.info("------New model instance created:------")

    name_model = str(model)

    logging.info("Model class: %s" %(name_model))     

    # model DS
    dsl = model_path.split('.')
    name=dsl[0]
    format_ds=dsl[1]

    ds_pk = name
    
    try:
        model_df = pd.DataFrame(model.coef_[0].reshape(1,len(dataframes[train].columns)), columns=list(dataframes[train].columns))

    except:
        model_df = pd.DataFrame(model.feature_importances_.reshape(1,len(dataframes[train].columns)), columns=list(dataframes[train].columns))

    
    fields = model_df.dtypes.to_dict()
    
    schema_field = [(el,str(fields[el])) for el in fields]

    schema_pk = name
    schemas[dsl[0]]=schema_pk
    schem_def[dsl[0]]=model_df.columns.to_list()
    dataframes[dsl[0]]=model_df
   
    logging.info("Model %s, saved as a %s and its schema is %s" %(name,format_ds,schema_field))
    
    report_link([train,test], dsl[0], False)   
  
    
    def flatten(d):
        out = {}
        for key, val in d.items():
            if isinstance(val, dict):
                val = [val]
            if isinstance(val, list):
                for subdict in val:
                    deeper = flatten(subdict).items()
                    out.update({key + '.' + key2: val2 for key2, val2 in deeper})
            else:
                out[key] = val
        return out

    from sklearn.metrics import classification_report


    predictions = model.predict(x_test)

    metrics = flatten(classification_report(y_test, predictions,output_dict=True))
    
    import json
    hyperparam = json.dumps(model.get_params())

    logging.info("Model %s, with those parameters %s, returns this classification report: %s" %(name,hyperparam,metrics))
    
    