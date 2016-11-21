#encoding=utf8
"""
encapsulate cch api as web service via flask
This aims at a tiny demo for demonstration but not for realease version
"""
from flask import Flask, request, redirect, url_for, session, send_from_directory
import sys
another_cch_path = '/Users/xiabofei/Documents/cchdir'
sys.path.append(another_cch_path)
import src as ix
# import cch as ix
from cch.envs import peek
import pandas as pd
import numpy as np
import os
from os.path import join as pjoin
P = lambda p: pjoin(base_dir, p)
base_dir = '/Users/xiabofei/Documents/cchdir/notebook/data'
tmp_dir = '/Users/xiabofei/Documents/cchdir/encapsulate/dc_data/'
from flask_wtf import Form
from wtforms import StringField, SubmitField, SelectField, IntegerField, SelectMultipleField
from wtforms.validators import DataRequired
from flask_bootstrap import Bootstrap
from flask import render_template
from flask_moment import Moment
from ipdb import set_trace as st
import pylab

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
app.config['UPLOAD_FOLDER'] = tmp_dir
bootstrap = Bootstrap(app)
moment = Moment(app)

@app.route('/uploads/<path:filename>')
def download_file(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/')
def index():
    return render_template('dc_default.html')

# 从DataSource读取文件
class ReadDataForm(Form):
    file_list = StringField(u'输入文件源', default=u'2010.csv;2009.csv;2008_2010.csv', validators=[DataRequired()])
    nrows_list = StringField(u'读入行数', default=u'1000;1000;1000', validators=[DataRequired()])
    submit = SubmitField(u'提交')
@app.route('/dc-dataset-register', methods=['GET', 'POST'])
def dc_dataset_register():
    paras = {}
    form = ReadDataForm()
    paras['form'] = form
    if request.method=='POST' and form.validate():
        paras['file_list'] = filter(None, form.file_list.data.strip().split(';'))
        paras['nrows_list'] = filter(None, form.nrows_list.data.strip().split(';'))
        try:
            # 保证长度与内容相等
            assert len(paras['file_list'])==len(paras['nrows_list']), u'file_list与nrows_list长度不相等'
            df_l = [
                    ( 'df_'+str(f.split('.')[0]), ix.read_csv_file_to_df(P(f), sep=',', nrows=int(n)) ) if f else None 
                    for (f,n) in zip(paras['file_list'], paras['nrows_list'])
                    ]
            # serilize to local filesystem
            for df in df_l: df[1].to_pickle(tmp_dir+str(df[0]))
            paras['df_l'] = df_l
            return render_template('dc_dataset_register.html', **paras)
        except Exception,e:
            return render_template('dc_error.html', e_message=e)
    return render_template('dc_dataset_register.html', **paras)

# 读取dataFrame并判断数据类型
class DatExpForm(Form):
    submit = SubmitField(u'提交')
@app.route('/dc-data-exploration', methods=['GET', 'POST'])
def dc_data_exploration():
    paras = {}
    form = DatExpForm()
    # 读取存在tmp_dir中的pickle数据(以后可改进为从database中读取)
    pickle_file_list = os.listdir(tmp_dir)
    paras['form'] = form
    paras['pickle_file_list'] = pickle_file_list
    ix.config.g_conf['settings_datatypemaxfactornum'] = 1000
    df_list = [ (f_name, pd.read_pickle(tmp_dir+f_name)) for f_name in pickle_file_list ]
    for df in df_list:
        ix.tag_meta_auto(df[1], num2factor_threshold=10)
    # 传入的meta_list (name, 2Dtable)
    meta_list = [ (df[0], trans_peek_to_2Dtable(peek(df[1], meta=True), df[1])) for df in df_list ]
    paras['meta_list'] = meta_list
    if request.method=='POST' and form.validate():
        try:
            return render_template('dc_data_exploration.html', **paras)
        except Exception,e:
            return render_template('dc_error.html', e_message=e)
    return render_template('dc_data_exploration.html', **paras)

def trans_peek_to_2Dtable(df_peek, df_ori):
    """
    将peek返回的dataframe进行处理 
    返回内容为 dict( meta_list, (columns_name, 2Dtable) )
    """
    ret = []
    meta_list = []
    meta_from_peek = ['col_name', 'col_datatype']
    meta_list += meta_from_peek + ['value_range', 'missing_value_percentage']
    meta_show_name_map = {'col_name':u'字段名称',
                          'col_datatype':u'字段类型',
                          'value_range':u'取值范围',
                          'missing_value_percentage':u'缺失值比例'}
    avaiable_dtypes = ['character', 'numeric', 'factor_s', 'empty', 'binary', 'datetime', 'factor_m']
    for col in df_peek.columns:
        tmp = [ df_peek[col][meta] for meta in meta_from_peek ]
        # Value Range
        dtype = df_peek[col]['col_datatype']
        if dtype not in avaiable_dtypes:
            tmp.append(u'uncertain data type')
        elif dtype =='character':
            tmp.append('character')
        elif dtype=='numeric':
            tmp.append( (df_ori[col].min(), df_ori[col].max()) )
            ax = df_ori[col].plot.hist()
            fig = ax.get_figure()
            fig.savefig(str(col)+'.png')
            fig.clf()
        elif dtype=='factor_s':
            tmp.append( df_ori[col].unique() )
        elif dtype=='empty':
            tmp.append( 'empty' )
        else:
            tmp.append(u'uncertain data type')
        # Missing Value Percentage
        tmp.append( str(df_ori[col].isnull().sum()*100.0 / len(df_ori[col]))+"%")
        ret.append(tmp)
    assert len(meta_list)==len(ret[0]), "meta data table columns not match"
    # map字段显示的名称
    meta_list = [ meta_show_name_map.get(meta, u'未设置显示名称') for meta in meta_list ]
    return (meta_list, ret)

if __name__ == '__main__':
    app.run()
