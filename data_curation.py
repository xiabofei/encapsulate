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
MD5_FILE = '/Users/xiabofei/Documents/cchdir/encapsulate/DATAFRAME_MD5'
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
# from time import sleep
import pylab
import glob
import hashlib
import json

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
                # 并计算每个dataframe的md5值 并存入数据表中
                with pd.HDFStore(HDF5_path) as store:
                    df_md5_tmp = {}
                    for df in df_l:
                        md5 = calculate_dataframe_md5(df[1])
                        df_name = df[0]+DATA_SUFF
                        df_md5_tmp[df_name] = md5
                        store.put(df_name, df[1])
                    # 与原有的(df_name, md5)进行merge
                    merge_dataframe_md5(df_md5_tmp)
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

def calculate_dataframe_md5(df):
    """
    功能
        计算dataframe的md5的值
    入参
        df : 需要计算md5的dataframe的instance
    """
    buf = df.to_csv()
    return hashlib.md5(buf).hexdigest()

def merge_dataframe_md5(df_md5_tmp):
    """
    功能
        更新dataframe对应的md5值并与本地之前存储的md5进行融合
    入参
        df_md5_tmp : 需要被融合的dataframe md5
    """
    with open(MD5_FILE,'w+') as f:
        df_md5 = {}
        if os.path.isfile(MD5_FILE) and os.stat(MD5_FILE).st_size!=0:
            df_md5 = json.load(f)
        for d,m in df_md5_tmp.items():
            df_md5[d] = m
        json.dump(df_md5, f)

def if_dataframe_md5_diff(df_name, df):
    """
    功能
        通过比较md5, 检查dataframe是否变化
    入参
        df_name : 从原有记录中找md5值的key
        df : dataframe的instance
    出参
        boolean : True则表明dataframe有更新 False表示dataframe没有更新
    """
    md5_new = calculate_dataframe_md5(df)
    with open(MD5_FILE,'w+') as f:
        df_md5 = {}
        if os.path.isfile(MD5_FILE) and os.stat(MD5_FILE).st_size!=0:
            df_md5 = json.load(f)
        if df_name not in df_md5.keys():
            df_md5[df_name] = md5_new 
            json.dump(df_md5, f)
            return True
        else:
            md5_old = df_md5[df_name]
            # dataframe没有改变
            if md5_old==md5_new:
                json.dump(df_md5, f)
                return False
            else:
                df_md5[df_name] = md5_new 
                json.dump(df_md5, f)
                return True

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
            # 验证df_list的md5信息是否有改变
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
    # 如果是get请求 也在div中加载相应的内容
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
    meta_from_peek = ['col_name', 'col_datatype', 'col_factornum']
    meta_list = meta_from_peek + ['value_range', 'missing_value_percentage']
    meta_show_name_map = {'col_name':u'字段名称',
                          'col_datatype':u'字段类型',
                          'col_factornum':u'区分度',
                          'distinct_ratio':u'区分度',
                          'value_range':u'取值范围',
                          'missing_value_percentage':u'缺失值比例'}
    for col_name in df_peek.columns:
        # 需要从peek中抽取的列信息
        tmp = []
        for meta in meta_from_peek:
            if meta=='col_factornum':
                distinct_ratio = int(df_peek[col_name][meta]) / (float(len(df_ori.index))+1)
                tmp.append(round(distinct_ratio,2))
            else:
                tmp.append(df_peek[col_name][meta])
        # tmp = [ df_peek[col_name][meta] for meta in meta_from_peek ]
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
        # 存放meta的dataframe中的col_name变成distinct_ratio
        meta_list = [ m if m!='col_factornum' else 'distinct_ratio' for m in meta_list ]
        df_col_meta = pd.DataFrame(columns=meta_list)
        for i in range(len(meta_list_write_to_local)):
            df_col_meta.loc[i] = meta_list_write_to_local[i]
        # 将该df列的信息存入HDF5中
        with pd.HDFStore(HDF5_path) as store:
            store.put(df_name+META_SUFF, df_col_meta)
    except Exception,e:
        print e
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

## 基础过滤涉及到的配置变量
# 是否处理该列的checkbox的判断标志
app.config['CHECKED'] = u'on'
# 记录对列指定操作的解释
app.config['COL_ACTION'] = { 
                            'DEL' : u'删除掉指定列',
                            'REMAIN' : u'保留指定列'
                            }
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
                    deal_base_column_filtering(store, df_name, col_name, 'DEL')
            # 更新显示meta信息
            paras['meta_list'] = get_df_meta_info(store) 
            store.close()
            return render_template('dc_data_cleansing.html', **paras)
        except Exception,e:
            return render_template('dc_error.html', e_message=e)
    paras['meta_list'] = get_df_meta_info(store) 
    store.close()
    return render_template('dc_data_cleansing.html', **paras)

# 后端需要从ajax接收的参数
col_filter_condition_paras = {
                              'data_type':'str',         # 保留该类型的列数据
                              'col_name_pattern':'str',  # 保留列名符合要求的列数据
                              'col_value_expr':'expr',    # 正则表达式根据列的值过滤
                              'distinct_ratio':'float',    # 根据列的区分度过滤
                              'non_NA_ratio':'float',      # 根据列数据非空率
                              'balance_ratio':'float'      # 列数据平衡率
                              }
@app.route('/dc-pro-data-cleansing', methods=['GET'])
def dc_pro_data_cleansing():
    """
    处理高级列过滤
    """
    paras = {}
    try:
        # 确认ajax传回的df_name存在于HDFStore中
        store = pd.HDFStore(HDF5_path)
        df_name = request.args.get('df_name', None)
        assert HDF5_PREF+df_name+DATA_SUFF in store.keys(), \
            "dataframe %s not in store %s"%(HDF5_PREF+df_name+DATA_SUFF, store.filename)
        # 接受ajax传回的传入select_columns_by_condition参数
        received_condition = {}
        for k in col_filter_condition_paras.keys():
            if request.args.get(k)!='':
                received_condition[k] = request.args.get(k)
        # 验证ajax传入的参数
        assert validate_pro_data_cleansing(received_condition), "recived paras from ajax are not valid"
        # 执行过滤模块 返回过滤后剩下的列名
        df = store[df_name+DATA_SUFF]
        ix.tag_meta_auto(df)
        remained_col_names = ix.select_columns_by_condition(df,**received_condition)
        # 更新HDFStore中的数据(DATA数据 META数据) 默认DATA数据和META数据都存在
        deal_pro_column_filtering(store, df_name, remained_col_names, 'REMAIN')
        # 读取出更新后的values 
        meta_table = store.get(hdf5_pref+df_name+meta_suff)
        paras['meta_table'] = meta_table
        store.close()
        return render_template('dc_pro_data_cleansing.html', **paras)
    except Exception,e:
        return render_template('dc_error.html', e_message=e)

def validate_pro_data_cleansing(received_condition):
    """
    功能
        验证ajax传入的列过滤参数有效性
        将部分传入字符转换为数值
    """
    try:
        for key in received_condition:
            key_type = col_filter_condition_paras.get(key)
            if key_type=='int':
                received_condition[key] = int(received_condition[key])
            elif key_type=='float':
                received_condition[key] = float(received_condition[key])
            else:
                continue
        return True
    except Exception,e:
        return False
    return False

def deal_pro_column_filtering(store, df_name, remained_col_names, action):
    """
    功能
        将列过滤的高级结果同步到HDFStore中
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
    for col_name in remained_col_names:
        assert col_name in store[df_name+DATA_SUFF].columns, \
                "column %s not in dataframe %s"%(col_name, df_name+DATA_SUFF)
    # 更新HDFStore中的数据
    store[df_name+DATA_SUFF] = store[df_name+DATA_SUFF][remained_col_names]
    # 删除列的meta信息 这里需要明确处理的对象是dataframe
    store[df_name+META_SUFF] = store[df_name+META_SUFF].loc[store[df_name+META_SUFF]['col_name'].isin(remained_col_names)]

def get_df_meta_info(store, col_datatype='ALL'):
    """
    功能
        从store中读取并更新meta信息
    入参
        store : 关联到hdf5的连接
        data_type : 需要查询的数据类型('ALL'为全体类型, 否则必须是avaiable_dtypes中的一种类型)
    出参
        meta_list : 返回最新的meta_info(dataframe名, dataframe实例)
    """
    meta_list = []
    if col_datatype=='ALL':
        for s_k in store.keys():
            if s_k.endswith(META_SUFF) and isinstance(store.get(s_k), pd.DataFrame): 
                meta_list.append( (extract_dataframe_name(s_k, '/', META_SUFF), store.get(s_k)) )
    else:
        assert col_datatype in avaiable_dtypes, u"datatype %s is not in available type"%(col_datatype)
        for s_k in store.keys():
            if s_k.endswith(META_SUFF) and isinstance(store.get(s_k), pd.DataFrame): 
                df = store.get(s_k)
                df = df[df['col_datatype']==col_datatype]
                meta_list.append( (extract_dataframe_name(s_k, '/', META_SUFF), df) )
    return meta_list

def deal_base_column_filtering(store, df_name, col_name, action):
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


@app.route('/dc-feature-engineering', methods=['GET'])
def dc_feature_engineering():
    """
    处理特征转换
    """
    paras = {}
    store = pd.HDFStore(HDF5_path)
    paras['meta_list'] = get_df_meta_info(store, 'factor_s') 
    store.close()
    return render_template('dc_feature_engineering.html', **paras)

@app.route('/dc-feature-engineering/factor', methods=['GET'])
def dc_feature_engineering_factor():
    paras = {}
    try:
        # 确认ajax传回的df_name存在于HDFStore中
        store = pd.HDFStore(HDF5_path)
        df_name = request.args.get('df_name', None)
        assert HDF5_PREF+df_name+DATA_SUFF in store.keys(), \
            "dataframe %s not in store %s"%(HDF5_PREF+df_name+DATA_SUFF, store.filename)
        # 接受ajax传回的需要特征打散的factor变量
        received_feature = []
        for f in request.args.getlist('checked_feature',None):
            received_feature.append(f.split('.')[1])
            print f
        derive_prefix = request.args.get('derive_prefix')
        assert validate_feature_engineering_factor(received_feature), "recived paras from ajax are not valid"
        deal_feature_engineering(store, df_name, received_feature, 'REMAIN', derive_prefix)
        store.close()
        return "test"
    except Exception,e:
        return render_template('dc_error.html', e_message=e)

def validate_feature_engineering_factor(received_feature):
    """
    验证factor feature传入的参数的有效性
    """
    return True

def deal_feature_engineering(store, df_name, feature_list, action, derive_prefix):
    """
    功能
        将传入的feature进行特征打散
    入参
        store : 与HDF5存储的接口
        df_name : 在HDF5中需要处理的dataframe
        feature_list : 需要处理的列
        action : 对传入的feature_list进行的操作
        derive_prefix : 由factor类型变量衍生出新列的前缀名
    """
    assert action in app.config['COL_ACTION'].keys(), \
            "columns action not in app.config['COL_ACTION']"
    assert HDF5_PREF+df_name+DATA_SUFF in store.keys(), \
            "dataframe %s not in store %s"%(HDF5_PREF+df_name+DATA_SUFF, store.filename)
    for feature in feature_list:
        assert feature in store[df_name+DATA_SUFF].columns, \
                "column %s not in dataframe %s"%(feature, df_name+DATA_SUFF)
    df = store[df_name+DATA_SUFF]
    ix.tag_meta_auto(df)
    derived = []
    for feature in feature_list:
        t = ix.derive_columns_from_factor(df, feature, derive_prefix=derive_prefix)
        derived.append(t)
    df = pd.concat([df]+derived, axis=1, join_axes=[df.index])
    # 更新data信息
    store[df_name+DATA_SUFF] = df
    # 生成衍生信息后再调用一次data_exploration生成meta信息

if __name__ == '__main__':
    app.run()
