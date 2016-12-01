#encoding=utf8
"""
encapsulate cch api as web service via flask
This aims at a tiny demo for demonstration but not for realease version
"""
from flask import Flask, request, redirect, url_for, session, send_from_directory, jsonify
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
base_dir = '/Users/xiabofei/Documents/cchdir/notebook/data/'
tmp_dir = '/Users/xiabofei/Documents/cchdir/encapsulate/dc_data/'
fig_dir = '/Users/xiabofei/Documents/cchdir/encapsulate/fig_data/'
meta_dir = '/Users/xiabofei/Documents/cchdir/encapsulate/meta_data/'
HDF5_path = '/Users/xiabofei/Documents/cchdir/encapsulate/data_curation.h5'
DATA_SUFF = '_DATA'
META_SUFF = '_META'
HDF5_PREF = '/'
DF_COL_SPLIT = '.'
ix.config.g_conf['settings_datatypemaxfactornum'] = 1000
from flask_wtf import Form
from wtforms import StringField, SubmitField, SelectField, IntegerField, SelectMultipleField
from wtforms.validators import DataRequired
from flask_bootstrap import Bootstrap
from flask import render_template
from flask_moment import Moment
from copy import copy, deepcopy
from pprint import pprint
from ipdb import set_trace as st
from time import sleep
import pylab
import glob

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
app.config['UPLOADS'] = {'tmp':tmp_dir, 'fig':fig_dir}
bootstrap = Bootstrap(app)
moment = Moment(app)

@app.route('/uploads/<directory>/<path:filename>')
def download_file(filename, directory='tmp'):
        return send_from_directory(app.config['UPLOADS'].get(directory), filename, as_attachment=True)

@app.route('/')
def index():
    return render_template('dc_default.html')

# 从DataSource读取文件
class ReadCSVForm(Form):
    file_list = StringField(u'输入文件源', default=u'2010.csv;2009.csv;2008_2010.csv', validators=[DataRequired()])
    nrows_list = StringField(u'读入行数', default=u'1000;1000;1000', validators=[DataRequired()])
    submit = SubmitField(u'提交')
# 从HDFStore文件中读数据
class ReadDFForm(Form):
    store_path = StringField(u'HDFStore文件', default=u'data_curation.h5', validators=[DataRequired()])
    submit = SubmitField(u'提交')
@app.route('/background_process')
def background_process():
    try:
        lang = request.args.get('proglang', 0, type=str)
        if lang.lower() == 'python':
            return jsonify(result='You are wise')
        else:
            return jsonify(result='Try again.')
    except Exception as e:
        return str(e)
@app.route('/dc-dataset-register', methods=['GET','POST'])
def dc_dataset_register():
    paras = {}
    csv_form = ReadCSVForm()
    df_form = ReadDFForm()
    paras['csv_form'] = csv_form 
    paras['df_form'] = df_form 
    if request.method=='POST':
        if request.form['submit']=='csv':
            paras['file_list'] = filter(None, csv_form.file_list.data.strip().split(';'))
            paras['nrows_list'] = filter(None,csv_form.nrows_list.data.strip().split(';'))
            try:
                # 保证长度与内容相等
                assert len(paras['file_list'])==len(paras['nrows_list']), u'file_list与nrows_list长度不相等'
                # df_l [(df名1, df1), (df名2, df2), ...]
                df_l = [
                        ( 'df_'+str(f.split('.')[0]), ix.read_csv_file_to_df(P(f), sep=',', nrows=int(n)) ) if f else None 
                        for (f,n) in zip(paras['file_list'], paras['nrows_list'])
                        ]
                # 以HDF5格式存入本地 ( 暂时用后缀区分data信息和meta信息 ) 
                with pd.HDFStore(HDF5_path) as store:
                    for df in df_l:
                        store.put(df[0]+DATA_SUFF, df[1])
                paras['df_l_from_csv'] = df_l
                paras['df_nrow'] = 20
                return render_template('dc_dataset_register.html', **paras)
            except Exception,e:
                return render_template('dc_error.html', e_message=e)
        elif  request.form['submit']=='df':
            # df_l [(df名1, df1), (df名2, df2), ...]
            df_l = []
            with pd.HDFStore(df_form.store_path.data.strip()) as store:
                for s_k in store.keys():
                    if isinstance(store.get(s_k), pd.DataFrame): 
                        if s_k.endswith(DATA_SUFF):
                            df_l.append( (extract_dataframe_name(s_k, HDF5_PREF, ''), store.get(s_k)) )
                        elif s_k.endswith(META_SUFF):
                            df_l.append( (extract_dataframe_name(s_k, HDF5_PREF, ''), store.get(s_k)) )
                        else:
                            pass
            paras['df_l_from_hdf5'] = df_l
            paras['df_nrow'] = 100
            return render_template('dc_dataset_register.html', **paras)
        else:
            return render_template('dc_dataset_register.html', **paras)
    return render_template('dc_dataset_register.html', **paras)

def extract_dataframe_name(name, pref, suff):
    """
    功能
        从name中扣除多余信息:前面的pref 后面的suff
    入参
        name : 从HDFStore中读出来的keys
        pref : 前面需要切掉的字符串
        suff : 后面需要切掉的字符串
    出参
        返回抽取后的dataframe的字符串
    """
    if pref=='' and suff!='':
        return name[:-len(suff)]
    elif pref!='' and suff=='':
        return name[len(pref):]
    elif pref!='' and suff!='':
        return name[len(pref):-len(suff)]
    else:
        return name

# 读取dataFrame并判断数据类型
@app.route('/dc-data-exploration', methods=['GET', 'POST'])
def dc_data_exploration():
    paras = {}
    if request.method=='POST':
        try:
            # 从HDF5中读取dataframe数据 并放入list array中
            df_list = []
            store = pd.HDFStore(HDF5_path)
            for df_name in store.keys():
                if df_name.endswith(DATA_SUFF) and isinstance(store.get(df_name), pd.DataFrame):
                    df_list.append([df_name, store.get(df_name)])
            # 生成meta信息(最主要的就是变量类型自动识别)
            for df in df_list:
                ix.tag_meta_auto(df[1], num2factor_threshold=10)
            # 传入的meta_list (name, 2Dtable)
            # 在处理dataframe meta信息时 处理df_name的名称
            for i in range(len(df_list)): df_list[i][0] = extract_dataframe_name(df_list[i][0], '/', DATA_SUFF) 
            # 这条语句不要改
            meta_list = [ (df[0], trans_peek_to_2Dtable(peek(df[1], meta=True), df[1], df[0])) for df in df_list ]
            paras['meta_list'] = meta_list
            store.close()
            return render_template('dc_data_exploration_content.html', **paras)
        except Exception,e:
            return render_template('dc_error.html', e_message=e)
    return render_template('dc_data_exploration.html', **paras)

# cch包可识别的数据类型
avaiable_dtypes = ['character', 'numeric', 'factor_s', 'empty', 'binary', 'datetime', 'factor_m']

def trans_peek_to_2Dtable(df_peek, df_ori, df_name):
    """
    说明
        将peek返回的df进行加工处理,
        将dataframe列的meta信息存下来
        将列的分布图的信息存在本地文件夹fig_dir中
    入参
        df_peek : cch包对df进行数据探索后的信息
        df_ori : 原始df
        df_name : df的名称 
    出参
        (meta_list, 2Dtable)
        meta_list : 表格每个字段对应显示名称 
        2Dtable : 各个表格的信息  
    """
    ret = []
    meta_list_write_to_local = []
    meta_from_peek = ['col_name', 'col_datatype']
    meta_list = meta_from_peek + ['value_range', 'missing_value_percentage']
    meta_show_name_map = {'col_name':u'字段名称',
                          'col_datatype':u'字段类型',
                          'value_range':u'取值范围',
                          'missing_value_percentage':u'缺失值比例'}
    for col_name in df_peek.columns:
        # 需要从peek中抽取的列信息
        tmp = [ df_peek[col_name][meta] for meta in meta_from_peek ]
        # 列变量取值范围
        dtype = df_peek[col_name]['col_datatype']
        if dtype not in avaiable_dtypes:
            tmp.append(u'uncertain data type')
        elif dtype =='character':
            tmp.append('character')
        elif dtype=='numeric':
            tmp.append( (df_ori[col_name].min(), df_ori[col_name].max()) )
        elif dtype=='factor_s':
            tmp.append( df_ori[col_name].unique() )
        elif dtype=='empty':
            tmp.append( 'empty' )
        else:
            tmp.append(u'uncertain data type')
        # 缺失值比例
        tmp.append( str(df_ori[col_name].isnull().sum()*100.0 / len(df_ori[col_name]))+"%")
        # 记录到meta list里面
        meta_list_write_to_local.append(deepcopy(tmp))
        # 生成fig信息
        #   这一部分要保证放在tmp的最后 因为不属于每一行共有的列的值
        #   在使用前加一个保护判断, 保证每行共有的列meta_list都处理完毕
        if len(tmp)==len(meta_list):
            draw_fig_for_df_oneCol(df_ori, df_name, col_name, dtype, tmp)
        ret.append(tmp)
    # assert len(meta_list)==len(ret[0]), "meta data table columns not match"
    try:
        # 将(不包含分布图)信息存入dataframe中
        df_col_meta = pd.DataFrame(columns=meta_list)
        for i in range(len(meta_list_write_to_local)):
            df_col_meta.loc[i] = meta_list_write_to_local[i]
        # 将该df列的信息存入HDF5中
        with pd.HDFStore(HDF5_path) as store:
            store.put(df_name+META_SUFF, df_col_meta)
    except Exception,e:
        from ipdb import set_trace as st
        st(context=21)
    finally:
        pass
    # 更新列字段显示名称
    meta_list = [ meta_show_name_map.get(meta, u'未设置显示名称') for meta in meta_list ]
    return (meta_list, ret)

# 定义'df列类型'到'分布图'的映射关系: 一个数据类型可以对应多个分布图
map_dtype_figtype = {'character' : [],
                     'numeric' : ['hist','box'],
                     'factor_s' : ['pie','bar'],
                     'empty' : [],
                     'binary' : [],
                     'datetime' : [], 
                     'factor_m' : []}
# 针对枚举类型数据
need_value_count_figtype = ['pie', 'bar']

def draw_fig_for_df_oneCol(df, df_name, col_name, dtype, tmp):
    """
    说明
        根据df某一列的数据类型生成分布图
    入参
        df : df信息
        df_name : 代表df名称的字符
        col_name : df中某一列的信息
        dtype : 该列的数据类型
        tmp :  [ (fig类型, fig地址)... ] 存放该列信息绑定的fig信息
    """
    for figtype in map_dtype_figtype.get(dtype):
        try:
            # ax = getattr(df[col_name].plot,figtype)()
            ax = df[col_name].plot(kind=figtype) if figtype not in need_value_count_figtype \
                    else df[col_name].value_counts().plot(kind=figtype)
            fig = ax.get_figure()
            fig_name = str(df_name)+'_'+str(col_name)+'_'+str(figtype)+'.png'
            fig.savefig(fig_dir+fig_name)
            print fig_name
            tmp.append((figtype, fig_name))
            fig.clf()
        except Exception,e:
            st(context=21)
        finally:
            pass

# 是否处理该列的checkbox的判断标志
app.config['CHECKED'] = u'on'
# 记录对列指定操作的解释
app.config['COL_ACTION'] = {'DEL':u'删除掉指定列'}
@app.route('/dc-data-cleansing', methods=['GET', 'POST'])
def dc_data_cleansing():
    paras = {}
    store = pd.HDFStore(HDF5_path)
    if request.method=='POST':
        try:
            # 遍历每一条需要更新的列信息 根据每一列的checkbox的值判断是否过滤掉这一列
            for rf_k in request.form.keys():
                # 根据选中的列过滤数据
                if request.form.getlist(rf_k)[0] == app.config['CHECKED']:
                    pprint(request.form.getlist(rf_k))
                    df_name = rf_k.split(DF_COL_SPLIT)[0]
                    col_name = rf_k.split(DF_COL_SPLIT)[1]
                    deal_with_column_filtering(store, df_name, col_name, 'DEL')
            # 更新显示meta信息
            paras['meta_list'] = get_df_meta_info(store) 
            store.close()
            return render_template('dc_data_cleansing.html', **paras)
        except Exception,e:
            return render_template('dc_error.html', e_message=e)
    paras['meta_list'] = get_df_meta_info(store) 
    store.close()
    return render_template('dc_data_cleansing.html', **paras)

def get_df_meta_info(store):
    """
    功能
        从store中读取并更新meta信息
    入参
        store : 关联到hdf5的连接
    出参
        meta_list : 返回最新的meta_info
    """
    meta_list = []
    for s_k in store.keys():
        if s_k.endswith(META_SUFF) and isinstance(store.get(s_k), pd.DataFrame): 
            meta_list.append( (extract_dataframe_name(s_k, '/', META_SUFF), store.get(s_k).values) )
    return meta_list

def deal_with_column_filtering(store, df_name, col_name, action):
    """
    功能
        判断df_name+DATA_SUFF是否在store中存在
        根据dc_data_cleansing页面form提交的信息, 将对应的dataframe进行处理
    入参
        store : 与HDF5存储的接口
        df_name : 在HDF5中需要处理的dataframe
        col_name : 需要处理的列名
        action : 需要对列进行的操作
    """
    assert action in app.config['COL_ACTION'].keys(), \
            "columns filtering action not in app.config['COL_ACTION']"
    assert HDF5_PREF+df_name+DATA_SUFF in store.keys(), \
            "dataframe %s not in store %s"%(HDF5_PREF+df_name+DATA_SUFF, store.filename)
    assert col_name in store[df_name+DATA_SUFF].columns, \
            "column %s not in dataframe %s"%(col_name, df_name+DATA_SUFF)
    # 删除实际列数据
    store[df_name+DATA_SUFF] = store[df_name+DATA_SUFF].drop(col_name,1)
    # 删除列的meta信息 
    store[df_name+META_SUFF] = store[df_name+META_SUFF][store[df_name+META_SUFF]['col_name']!=col_name]

if __name__ == '__main__':
    app.run()
