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
import pylab
import glob
import hashlib
import json
import urllib2

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
                    # merge_dataframe_md5(df_md5_tmp)
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

@app.route('/dc-delete-dataframe', methods=['GET'])
def delete_dataframe():
    """
    功能
        删除dataframe
        这部分接受ajax的df_name_type参数是带有DATA或META后缀的
            1) 如果删除的是DATA_SUFF后缀 则同步删除相应的META数据
            2) 如果删除的是META_SUFF后缀 则不用同步删除相应的DATA数据
    """
    try:
        store = pd.HDFStore(HDF5_path)
        df_name_type = request.args.get('df_name_type', None)
        assert HDF5_PREF+df_name_type in store.keys(), \
            "dataframe %s not in store %s"%(HDF5_PREF+df_name_type, store.filename)
        store.remove(HDF5_PREF+df_name_type)
        if df_name_type.endswith(DATA_SUFF):
            potential_meta_name = HDF5_PREF+extract_dataframe_name(df_name_type,'',DATA_SUFF)+META_SUFF
            if potential_meta_name in store.keys():
                store.remove(potential_meta_name)
        store.close()
        return redirect(url_for('dc_dataset_register'))
    except Exception,e:
        return render_template('dc_error.html', e_message=e)

@app.route('/store-dataframe-to-csv', methods=['GET'])
def store_dataframe_to_csv():
    """
    将dataframe另存为csv格式
    """
    try:
        st(context=21)
        store = pd.HDFStore(HDF5_path)
        df_name_type = request.args.get('df_name_type', None)
        assert HDF5_PREF+df_name_type in store.keys(), \
            "dataframe %s not in store %s"%(HDF5_PREF+df_name_type, store.filename)
        df = store[HDF5_PREF+df_name_type]
        df.to_csv('/Users/xiabofei/Documents/cchdir/notebook/data/'+str(df_name_type)+'.csv', index=False)
        return redirect(url_for('dc_dataset_register'))
    except Exception,e:
        return render_template('dc_error.html', e_message=e)

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
        df_md5_tmp : 需要被融合的{dataframe:md5,...} 
    """
    df_md5 = {}
    with open(MD5_FILE,'r') as f:
        if os.path.isfile(MD5_FILE) and os.stat(MD5_FILE).st_size!=0:
            df_md5 = json.load(f)
    with open(MD5_FILE,'w') as f:
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
    # 获取原有的dataframe与md5
    df_md5 = {}
    if os.path.isfile(MD5_FILE) and os.stat(MD5_FILE).st_size!=0:
        f = open(MD5_FILE,'r+')
        df_md5 = json.load(f)
        f.close()
    with open(MD5_FILE,'w+') as f:
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
            # 0.推给前端数据结构为meta_list
            #   [(df_name1, (meta_column_names1, meta_data1)), (df_name2, (meta_column_names2, meta_data2)), ... ]
            #   这里的df_name是干净的不带前后缀的
            meta_list = []
            # 1.从HDFStore中读取dataframe 并分流为需要exploration和不需要exploration的部分
            store = pd.HDFStore(HDF5_path)
            df_list_need_exploration = []
            df_list_not_exploration = []
            divide_dataframe_for_exploration(store, df_list_need_exploration, df_list_not_exploration)
            # 2.处理需要exploration的dataframe
            for df in df_list_need_exploration:
                ix.tag_meta_auto(df[1], num2factor_threshold=2)
                # 重写dataframe对应的meta的全表信息 & 信息添加到meta_list中
                meta_column_names, meta_df = trans_peek_to_2Dtable(peek(df[1], meta=True), df[1], df[0]) 
                meta_column_names = [ meta_show_name_map.get(meta, u'未设置显示名称') for meta in meta_column_names ]
                meta_list.append( (df[0], (meta_column_names, meta_df.values)) )
            # 3.处理不需要exploration的dataframe
            for df in df_list_not_exploration:
                meta_df = store.get(df[0]+META_SUFF)
                meta_column_names = meta_df.columns
                meta_column_names = [ meta_show_name_map.get(meta, u'未设置显示名称') for meta in meta_column_names ]
                meta_list.append( (df[0], (meta_column_names, meta_df.values)) )
            store.close()
            # 4.推给前端meta_list #
            paras['meta_list'] = meta_list
            return render_template('dc_data_exploration_content.html', **paras)
        except Exception,e:
            return render_template('dc_error.html', e_message=e)
    return render_template('dc_data_exploration.html', **paras)

def divide_dataframe_for_exploration(store, df_list_need_exploration, df_list_not_exploration):
    """
    功能
        判断store中哪些dataframe需要被exploration,判断准则:
            准则1. dataframe的md5值与原有的不同
            准则2. dataframe没有对应的meta data
            准则3. dataframe的列数量与meta data表中的不同
            准则4. meta表对应的列数与要求的meta_names数目不同
    入参
        store : 与HDFStore的连接
        df_list_need_exploration : list 记录需要被exploration的dataframe
        df_list_not_exploration : list 记录不需要被exploration的dataframe
    """
    for df_name in store.keys():
        if df_name.endswith(DATA_SUFF) and isinstance(store.get(df_name), pd.DataFrame):
            df = store.get(df_name)
            df_name = extract_dataframe_name(df_name, '/', DATA_SUFF)
            # 准则1
            if if_dataframe_md5_diff(df_name+DATA_SUFF, df):
                df_list_need_exploration.append([df_name, df])
            # 准则2 准则3
            elif if_dataframe_meta_not_match(store, df_name, df):
                df_list_need_exploration.append([df_name, df])
            # 准则4
            elif len(store.get(HDF5_PREF+df_name+META_SUFF).columns)!=len(meta_names):
                df_list_need_exploration.append([df_name, df])
            else:
                df_list_not_exploration.append([df_name, df])

def if_dataframe_meta_not_match(store, df_name, df):
    """
    功能
        分析dataframe与对应的meata data是否匹配
        作为判断dataframe是否需要exploration的依据
    入参
        store : 与HDFStore的连接
        df_name : dataframe的裸名
        df : dataframe本身
    """
    # 1.没有对应的meta表
    if HDF5_PREF+df_name+META_SUFF not in store.keys():
        return True
    # 2.meta表中的columns数与dataframe的column不同
    elif len(df.columns)!=len(store.get(df_name+META_SUFF).index):
        return True
    else:
        return False

# cch包可识别的数据类型
avaiable_dtypes = ['character', 'numeric', 'factor_s', 'empty', 'binary', 'datetime', 'factor_m']
# meta信息表对应的列名的显示信息
meta_show_name_map = {'col_name':u'字段名称',
                      'col_datatype':u'字段类型',
                      'col_factornum':u'区分度',
                      'distinct_ratio':u'区分度',
                      'value_range':u'取值范围',
                      'missing_value_percentage':u'缺失值比例',
                      'distribution_figs':u'详细信息'}
meta_from_peek = ['col_name', 'col_datatype', 'col_factornum']
meta_names = meta_from_peek + ['value_range', 'missing_value_percentage','distribution_figs']

def trans_peek_to_2Dtable(df_peek, df_ori, df_name):
    """
    说明
        将peek返回的df进行加工处理,
        将dataframe列的meta信息存下来
        将列的分布图的图像存在本地文件夹fig_dir中
        格式要求:
            1.保证col_name放在第一个column
            2.保证fig信息要放在最后一个column
    入参
        df_peek : cch包对df进行数据探索后的信息
        df_ori : 原始df
        df_name : df的名称 
    出参
        (col_names, dataframe)
        col_names : 表格每个字段对应显示名称 
        dataframe : 各个表格的信息  
    """
    meta_array = []
    col_names = []
    # 构造需要写入dataframe对应的meta表的每一列的信息
    for col_name in df_peek.columns:
        # 0.直接从peek返回信息中抽取的内容
        tmp = []
        for meta in meta_from_peek:
            # peek中返回的'col_factornum'的名称是写死的不能修改 根据该列的信息可以计算出distinct_ratio的数据
            # 在这里先做蹩脚的转换
            if meta=='col_factornum':
                distinct_ratio = int(df_peek[col_name][meta]) / (float(len(df_ori.index))+1)
                tmp.append(round(distinct_ratio,2))
            else:
                tmp.append(df_peek[col_name][meta])
        # 1.列变量取值范围
        dtype = df_peek[col_name]['col_datatype']
        tmp.append( get_colrange_by_coltype(df_ori, col_name, dtype) )
        # 2.缺失值比例
        tmp.append( str(df_ori[col_name].isnull().sum()*100.0 / len(df_ori[col_name]))+"%")
        # 3.列变量分布信息
        tmp.append(create_distribution_fig(df_ori, df_name, col_name, dtype))
        meta_array.append(tmp)
    try:
        # 重写meta信息 全表
        # 存放meta的dataframe中的col_name变成distinct_ratio 原因还是peek中蹩脚的'col_factornum'的列名
        col_names = [ m if m!='col_factornum' else 'distinct_ratio' for m in meta_names ]
        # 生成dataframe存放meta信息
        df_col_meta = pd.DataFrame(columns=col_names)
        for i in range(len(meta_array)): df_col_meta.loc[i] = meta_array[i]
        # 将meta信息存入HDF5中
        # 由于这里需要把meta信息存入HDF5 有fig对应的值就有对应的列名 因此要修改前端显示的模板
        with pd.HDFStore(HDF5_path) as store:
            store.put(df_name+META_SUFF, df_col_meta)
    except Exception,e:
        print e
        st(context=21)
    return (col_names, df_col_meta)

def get_colrange_by_coltype(df_data, col_name, dtype):
    """
    功能
        根据列变量类型 获取其取值范围

    入参
        df_data : 待探索的原始datafraem
        col_name : 数据列名称
        dtype : 这一列的数据类型

    出参
        列的取值分布:
            如果是numeric, 则返回(min, max)
            如果是category, 则返回[c1, c2, c3, ...]
            如果是其他类型, 暂时不做处理, 只返回该类型名
    """
    if dtype not in avaiable_dtypes:
        return u'uncertain data type'
    elif dtype =='character':
        return u'character'
    elif dtype=='numeric':
        return (df_data[col_name].min(), df_data[col_name].max())
    elif dtype=='factor_s':
        return df_data[col_name].unique()
    elif dtype=='empty':
        return u'empty'
    else:
        return u'uncertain data type'

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

def create_distribution_fig(df, df_name, col_name, dtype):
    """
    说明
        根据df某一列的数据类型生成分布图
    入参
        df : df信息
        df_name : 代表df名称的字符
        col_name : df中某一列的信息
        dtype : 该列的数据类型
    出参
        将该列对应的distribution fig信息拼接成字符串
            figname1#figname2...
        如果没有信息则返回为空
    """
    ret = []
    for figtype in map_dtype_figtype.get(dtype):
        try:
            ax = df[col_name].plot(kind=figtype) if figtype not in need_value_count_figtype \
                    else df[col_name].value_counts().plot(kind=figtype)
            fig = ax.get_figure()
            fig_name = str(df_name)+'_'+str(col_name)+'_'+str(figtype)+'.png'
            fig.savefig(fig_dir+fig_name)
            print fig_name
            ret.append(fig_name)
            fig.clf()
        except Exception,e:
            st(context=21)
    return "#".join(ret)


## 基础过滤涉及到的配置变量
# 是否处理该列的checkbox的判断标志
app.config['CHECKED'] = u'on'
# 记录对列指定操作的解释
app.config['COL_ACTION'] = { 
                            'DEL' : u'删除掉指定列',
                            'REMAIN' : u'保留指定列',
                            'CREATE' : u'创建新dataframe'
                            }
@app.route('/dc-data-cleansing', methods=['GET', 'POST'])
def dc_data_cleansing():
    paras = {}
    store = pd.HDFStore(HDF5_path)
    if request.method=='POST':
        try:
            # 遍历每一条需要更新的列信息 根据每一列的checkbox的值判断是否过滤掉这一列
            df_name_list = []
            for rf_k in request.form.keys():
                # 根据选中的列过滤数据
                if request.form.getlist(rf_k)[0] == app.config['CHECKED']:
                    pprint(request.form.getlist(rf_k))
                    df_name = rf_k.split(DF_COL_SPLIT)[0]
                    col_name = rf_k.split(DF_COL_SPLIT)[1]
                    deal_base_column_filtering(store, df_name, col_name, 'DEL')
                    df_name_list.append(df_name)
            # 由于只是删除某列 不需要重新生成dataframe的meta信息 只需要重新计算md5并更新即可
            df_md5 = {}
            for df_name in list(set(df_name_list)):
                df_md5[df_name+DATA_SUFF] = calculate_dataframe_md5(store[df_name+DATA_SUFF])
                merge_dataframe_md5(df_md5)
            # 更新显示meta信息
            paras['meta_list'] = get_df_meta_info(store) 
            store.close()
            return render_template('dc_data_cleansing.html', **paras)
        except Exception,e:
            return render_template('dc_error.html', e_message=e)
    paras['meta_list'] = get_df_meta_info(store) 
    store.close()
    return render_template('dc_data_cleansing.html', **paras)

@app.route('/dc-filter-row-by-id', methods=['GET'])
def df_filter_row_by_id():
    """
    功能
        根据A表提供的ID 对B表的ID进行过滤
    """
    st(context=21)
    ret = {}
    try:
        # 获取相应数据
        store = pd.HDFStore(HDF5_path)
        df_name = request.args.get('origin_samples', None)
        assert HDF5_PREF+df_name+DATA_SUFF in store.keys(), \
            "dataframe %s not in store %s"%(HDF5_PREF+df_name+DATA_SUFF, store.filename)
        df_id_name = request.args.get('id_candicate', None)
        assert HDF5_PREF+df_id_name+DATA_SUFF in store.keys(), \
            "dataframe %s not in store %s"%(HDF5_PREF+df_id_name+DATA_SUFF, store.filename)
        new_dataframe_name = request.args.get('new_dataframe_name', None)
        df = store[df_name+DATA_SUFF]
        ix.tag_meta_auto(df)
        tmp1 = store[df_id_name+DATA_SUFF]["DESYNPUF_ID"]
        def f1(*args):
            id = args[0]
            return id in [a for a in tmp1]
        cnt_rows = ix.select_rows_by_function(df, f1)
        df = ix.filter_rows(df, cnt_rows)
        update_df_in_HDFStore_by_name(store, new_dataframe_name, df)
        # assert df_name!=new_dataframe_name, "df_name and new_dataframe_name cannot be the same"
        # store.put(new_dataframe_name+DATA_SUFF, df)
        ret['info'] = 'affects %s number of rows'%(str(sum(cnt_rows)))
        store.close()
    except Exception, e:
        store.close()
        return render_template('dc_error.html', e_message=e)
    return json.dumps(ret) 

@app.route('/dc-select-row-by-expr', methods=['GET','POST'])
def dc_select_row_by_expr():
    """
    功能
        根据表达式过滤行
    """
    st(context=21)
    ret = {}
    try:
        store = pd.HDFStore(HDF5_path)
        # 接受ajax传回参数
        df_name = request.args.get('df_name', None)
        assert HDF5_PREF+df_name+DATA_SUFF in store.keys(), \
            "dataframe %s not in store %s"%(HDF5_PREF+df_name+DATA_SUFF, store.filename)
        new_dataframe_name = request.args.get('new_dataframe_name', None)
        rval_expr = request.args.get('rval_expr', None)
        # non_NA_percent = request.args.get('non_NA_percent',None)
        non_NA_percent = 0
        # expr_symbol = urllib2.unquote(request.args.get('expr_symbol', None))
        expr_symbol = '\$' 
        # 按条件过滤数据 
        df = store[df_name+DATA_SUFF]
        ix.tag_meta_auto(df)
        cnt_rows = ix.select_rows_by_expr(
                    df, 
                    expr_symbol=expr_symbol,
                    non_NA_percent=non_NA_percent,
                    rval_expr=rval_expr
                    )
        df = ix.filter_rows(df, cnt_rows)
        # 生成新dataframe 
        update_df_in_HDFStore_by_name(store, new_dataframe_name, df)
        ret['info'] = 'affects %s number of rows'%(str(sum(cnt_rows)))
        store.close()
    except Exception, e:
        store.close()
        return render_template('dc_error.html', e_message=e)
    return json.dumps(ret) 
        
        
def update_df_in_HDFStore_by_name(store, new_dataframe_name, df):
    """
    功能
        向HDFStore中存储新的dataframe时
            1) 如果HDFStore中原有的dataframe的name相同 则需要更新相应的meta data    
            2) 将dataframe存入HDFStore中
    入参
        store : 与HDFStore的连接
        new_dataframe_name : 新生成的dataframe的name
        df : 新生成的dataframe
    """
    if HDF5_PREF+new_dataframe_name+DATA_SUFF in store.keys():
        if HDF5_PREF+new_dataframe_name+META_SUFF in store.keys():
            store.remove(HDF5_PREF+new_dataframe_name+META_SUFF)
    store.put(new_dataframe_name+DATA_SUFF, df)
            

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
    功能
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
        # 通过字典col_filter_condition_paras来控制截取需要从ajax截取的参数
        received_condition = {}
        for k in col_filter_condition_paras.keys():
            if request.args.get(k)!='':
                received_condition[k] = request.args.get(k)
        # 验证ajax传入的参数
        assert validate_pro_data_cleansing(received_condition), "recived paras from ajax are not valid"
        # 执行过滤模块 返回过滤后剩下的列名
        df = store[df_name+DATA_SUFF]
        ix.tag_meta_auto(df)
        # A tmp tradeoff method to correct the datatype of all ready explored dataframe 
        assert HDF5_PREF+df_name+META_SUFF in store.keys(), \
            "dataframe %s not in store %s"%(HDF5_PREF+df_name+META_SUFF, store.filename)
        df_meta = store[HDF5_PREF+df_name+META_SUFF]
        for i in range(len(df_meta.index)):
            col_name = df_meta.ix[i]['col_name']
            col_datatype = df_meta.ix[i]['col_datatype']
            ix.update_meta(df, [col_name], "col_datatype", col_datatype)
        remained_col_names = ix.select_columns_by_condition(df,**received_condition)
        # 更新HDFStore中的数据(DATA数据 META数据) 默认DATA数据和META数据都存在
        # 1) 如果是'CREATE'模式 则生成新dataframe, 并传入新dataframe的name
        # 2) 如果是'REMAIN'模式 则更新原有dataframe并保留符合条件的列
        if request.args.get('new_dataframe_name','') != '':
                paras['meta_table'] = deal_pro_column_filtering(store, df_name, \
                        remained_col_names, 'CREATE', request.args.get('new_dataframe_name'))
        else:
            paras['meta_table'] = deal_pro_column_filtering(store, df_name, remained_col_names, 'REMAIN')
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

def deal_pro_column_filtering(store, df_name, col_names, action, new_dataframe_name='df_derived_newname'):
    """
    功能
        根据action对高级筛选的结果进行操作
    入参
        store : 与HDF5存储的接口
        df_name : 在HDF5中需要处理的dataframe
        col_name : 需要处理的列名
        action : 需要对列进行的操作
        new_dataframe_name : 如果是'CREATE'操作 则传入新生成的dataframe的变量名
    出参
        返回meta_table 用于填充页面显示的内容
    """
    assert action in app.config['COL_ACTION'].keys(), \
            "columns filtering action not in app.config['COL_ACTION']"
    assert HDF5_PREF+df_name+DATA_SUFF in store.keys(), \
            "dataframe %s not in store %s"%(HDF5_PREF+df_name+DATA_SUFF, store.filename)
    for col_name in col_names:
        assert col_name in store[df_name+DATA_SUFF].columns, \
                "column %s not in dataframe %s"%(col_name, df_name+DATA_SUFF)
    df_md5 = {}
    if action=='REMAIN':
        # 更新HDFStore中的数据
        store[df_name+DATA_SUFF] = store[df_name+DATA_SUFF][col_names]
        # 删除列的meta信息 这里需要明确处理的对象是dataframe
        store[df_name+META_SUFF] = store[df_name+META_SUFF].loc[store[df_name+META_SUFF]['col_name'].isin(col_names)]
        # 由于没有生成新的列 只需要更新dataframe的md5即可
        df_md5[df_name+DATA_SUFF] = calculate_dataframe_md5(store[df_name+DATA_SUFF])
        merge_dataframe_md5(df_md5)
        # 把meta_table推到前端页面
        return store.get(HDF5_PREF+df_name+META_SUFF)
    elif action=='CREATE':
        # 如果采用复杂的策略 : 则避免了这种问题 把meta的生成工作都放在了统一的部分执行
        # 选定采用复杂策略
        # 抽出data列
        new_df_data = store[df_name+DATA_SUFF][col_names]
        # 抽出meta列
        remain_meta = store[df_name+META_SUFF].loc[store[df_name+META_SUFF]['col_name'].isin(col_names)]
        # 只存data
        update_df_in_HDFStore_by_name(store, new_dataframe_name, new_df_data)
        # store.put(new_dataframe_name+DATA_SUFF, new_df_data)
        return remain_meta 
    else:
        raise ValueError("action value %s not validated",action)

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


def remain_col_from_meta_list(meta_list, col_name_list):
    """
    功能
        针对get_df_meta_info返回的meta_list中剔除某些不许要显示的列
    入参
        meta_list : get_df_meta_info返回的meta_list
        col_names : 需要显示的meta中的列名
    出参
        ret : [[df_name, df_data],...] 
    """
    ret = []
    for col_name in col_name_list:
        assert col_name in meta_names, "col_name '%s' not in meta table column "%(col_name)
    for meta in meta_list:
        df_name = meta[0]
        df_data = meta[1][col_name_list]
        ret.append([df_name, df_data])
    return ret



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
    store[df_name+DATA_SUFF] = store[df_name+DATA_SUFF].drop(col_name, 1)
    # 删除列的meta信息 
    store[df_name+META_SUFF] = store[df_name+META_SUFF][store[df_name+META_SUFF]['col_name']!=col_name]


@app.route('/dc-feature-engineering', methods=['GET'])
def dc_feature_engineering():
    """
    处理特征转换
    """
    try:
        paras = {}
        with pd.HDFStore(HDF5_path) as store:
            # 对特征处理时需要展现的列的信息
            col_name_list = ['col_name', 'col_datatype', 'value_range']
            paras['concat_dataframe'] = remain_col_from_meta_list(\
                    get_df_meta_info(store, 'ALL'),
                    col_name_list)
            paras['meta_list_factor_s'] = remain_col_from_meta_list( \
                    get_df_meta_info(store, 'factor_s'),
                    col_name_list)
            paras['meta_list_col_type_update'] = remain_col_from_meta_list( \
                    get_df_meta_info(store, 'ALL'),
                    col_name_list)
            paras['meta_list_datetime'] = remain_col_from_meta_list(\
                    get_df_meta_info(store, 'datetime'),
                    col_name_list)
            paras['meta_list_one_to_one'] = remain_col_from_meta_list(\
                    get_df_meta_info(store, 'ALL'),
                    col_name_list)
            return render_template('dc_feature_engineering.html', **paras)
    except Exception,e:
        return render_template('dc_error.html', e_message=e)

@app.route('/dc-concat-dataframe', methods=['GET'])
def dc_concat_dataframe():
    """
    功能
        接收ajax参数 对两个dataframe执行join/merge/concat等操作
        (第一版先满足打通基本流程 后面的版本再处理复杂情况)
    入参
        left_df_name : concat操作中的left dataframe名
        right_df_name : concat操作中的right dataframe名
        axis : 0代表按行合并, 1代表按列合并
        join_axes : 作为concat操作中的key列
        new_dataframe_name : 新生成的dataframe的名
    出参
        返回新生成的dataframe信息的json串
    """
    st(context=21)
    ret = {}
    with pd.HDFStore(HDF5_path) as store:
        left_df_name = request.args.get('left_df_name', None)
        assert HDF5_PREF+left_df_name+DATA_SUFF in store.keys(), \
                "dataframe %s not in store %s"%(HDF5_PREF+left_df_name+DATA_SUFF, store.filename)
        right_df_name = request.args.get('right_df_name', None)
        assert HDF5_PREF+right_df_name+DATA_SUFF in store.keys(), \
                "dataframe %s not in store %s"%(HDF5_PREF+right_df_name+DATA_SUFF, store.filename)
        axis = int(request.args.get('axis', None))
        join_axes = request.args.get('join_axes', None)
        how = request.args.get('how',None)
        new_dataframe_name = request.args.get('new_dataframe_name', None)
        left = store[HDF5_PREF+left_df_name+DATA_SUFF]
        right = store[HDF5_PREF+right_df_name+DATA_SUFF]
        # 临时将concat join merge合并在一起实现
        # 第二版将这三种情况分别实现
        if axis==1:
            new_dataframe = pd.concat([left,right], axis=axis, join_axes=[left.index])
        else:
            if join_axes=='':
                new_dataframe = pd.merge(left, right, left_index=True, right_index=True,suffixes=['_l', '_r'] )
            else:
                if how=='':
                    new_dataframe = ix.df_join(left, right, [join_axes])
                else:
                    new_dataframe = ix.df_join(left, right, [join_axes], how=how)
        update_df_in_HDFStore_by_name(store, new_dataframe_name, new_dataframe)
        ret['new_dataframe_name'] = new_dataframe_name
    return json.dumps(ret) 

@app.route('/get-meta-dataframe', methods=['GET'])
def get_meta_dataframe():
    """
    功能
        接收ajax推送的参数 返回相应数据的dataframe的data dimension和meta信息
    入参
        df_name : ajax传入的dataframe的名字
    出参
        paras : 
            paras.df_meta df_name代表的dataframe的meta信息
            paras.df_data df_name代表的dataframe的数据样例
            paras.shape df_data的shape信息
    """
    paras = {}
    try:
        store = pd.HDFStore(HDF5_path)
        df_name = request.args.get('df_name', None)
        assert HDF5_PREF+df_name+META_SUFF in store.keys(), \
            "dataframe %s not in store %s"%(HDF5_PREF+df_name+META_SUFF, store.filename)
        assert HDF5_PREF+df_name+DATA_SUFF in store.keys(), \
            "dataframe %s not in store %s"%(HDF5_PREF+df_name+DATA_SUFF, store.filename)
        df_meta = store[df_name+META_SUFF]
        df_data = store[df_name+DATA_SUFF]
        # step1. 处理df的meta信息 
        col_name_list = ['col_name', 'col_datatype', 'value_range']
        meta_list = remain_col_from_meta_list([[df_name,df_meta]], col_name_list)
        df_meta = meta_list[0][1]
        paras['df_meta'] = df_meta.to_html()
        # step2. 获取dataframe shape 
        paras['shape'] = "("+str(df_data.shape[0])+","+str(df_data.shape[1])+")"
        # step3. 获取dataframe shape 
        df_data = df_data.head(n=10)
        paras['df_data'] = df_data.to_html()
        store.close()
        return json.dumps(paras)
    except Exception,e:
        return render_template('dc_error.html', e_message=e)

@app.route('/dc-feature-engineering/coltype-update', methods=['GET'])
def dc_feature_engineering_coltype_update():
    paras = {}
    try:
        # 确认ajax传回的df_name存在于HDFStore中
        store = pd.HDFStore(HDF5_path)
        df_name = request.args.get('df_name', None)
        assert HDF5_PREF+df_name+DATA_SUFF in store.keys(), \
            "dataframe %s not in store %s"%(HDF5_PREF+df_name+DATA_SUFF, store.filename)
        assert HDF5_PREF+df_name+META_SUFF in store.keys(), \
            "dataframe %s not in store %s"%(HDF5_PREF+df_name+META_SUFF, store.filename)
        df_meta = store[df_name+META_SUFF]
        df_data = store[df_name+DATA_SUFF]
        # 解析需要进行特征类型转换的col_name和col_datatype
        for f in request.args.getlist('colname_coltype',None):
            col_name = f.split('.')[1]
            col_datatype = f.split('.')[2]
            df_meta.ix[df_meta[df_meta['col_name']==col_name].index.values[0],'col_datatype'] = col_datatype
            # 更新meta表中的distribution_figs
            distribution_figs = create_distribution_fig(df_data, df_name, col_name, col_datatype)
            df_meta.ix[df_meta[df_meta['col_name']==col_name].index.values[0],'distribution_figs'] = distribution_figs
            # 更新meta表中的value_range
            value_range = get_colrange_by_coltype(df_data, col_name, col_datatype)
            df_meta.ix[df_meta[df_meta['col_name']==col_name].index.values[0],'value_range'] = str(value_range)
        # 更新列的meta信息
        store[df_name+META_SUFF] = df_meta
        store.close()
        return redirect(url_for('dc_feature_engineering'))
    except Exception,e:
        return render_template('dc_error.html', e_message=e)

@app.route('/dc-feature-engineering/datetime', methods=['GET'])
def dc_feature_engineering_datetime():
    """
    功能
        能进入到这里的数据 都保证是datetime类型了
        将datetime列按格式转换, 并进行数值化处理
    """
    st(context=21)
    ret = {}
    try:
        store = pd.HDFStore(HDF5_path)
        df_name = request.args.get('df_name', None)
        new_dataframe_name = request.args.get('new_dataframe_name',None)
        value_as_base = pd.to_datetime(request.args.get('value_as_base'))
        derive_prefix = request.args.get('derive_prefix',None)
        assert HDF5_PREF+df_name+DATA_SUFF in store.keys(), \
            "dataframe %s not in store %s"%(HDF5_PREF+df_name+DATA_SUFF, store.filename)
        df = store[df_name+DATA_SUFF]
        col_name_list = []
        for col_name in request.args.getlist('col_names',None):
            col_name_list.append(col_name.split('.')[1])
            print col_name
        # 将选中的datetime类型的列进行格式转换
        ix.tag_meta_auto(df)
        ix.update_meta(df, col_name_list, "col_datatype","datetime")
        ix.type_casting(df, col_name_list, dt_format="%Y%m%d") 
        store[df_name+DATA_SUFF] = df
        # 将转换后的数据存入新dataframe中
        t_df = ix.derive_columns_from_datetime(
                df, 
                col_name_list, 
                value_as_base=value_as_base,
                inverse=True,
                derive_prefix=derive_prefix)
        if new_dataframe_name!='':
            update_df_in_HDFStore_by_name(store, new_dataframe_name, t_df) 
        # 这里不需要跟新md5 因为格式已经变化
        ret['impact_columns'] = str(len(col_name_list))
        store.close()
        return json.dumps(ret)
    except Exception,e:
        return render_template('dc_error.html', e_message=e)

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
        new_dataframe_name = request.args.get('new_dataframe_name')
        assert validate_feature_engineering_factor(received_feature), "recived paras from ajax are not valid"
        deal_feature_engineering(store, df_name, received_feature, 'REMAIN', derive_prefix, new_dataframe_name)
        store.close()
        return redirect(url_for('dc_feature_engineering'))
    except Exception,e:
        return render_template('dc_error.html', e_message=e)

def validate_feature_engineering_factor(received_feature):
    """
    验证factor feature传入的参数的有效性
    """
    return True

def deal_feature_engineering(store, df_name, feature_list, action, derive_prefix, new_dataframe_name):
    """
    功能
        1) 将传入的feature进行特征打散
        2) 将新生成的特征存入dataframe
    入参
        store : 与HDF5存储的接口
        df_name : 在HDF5中需要处理的dataframe
        feature_list : 需要处理的列
        action : 对传入的feature_list进行的操作
        derive_prefix : 由factor类型变量衍生出新列的前缀名
        new_dataframe_name : 新生成的dataframe的name 
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
    df = pd.concat(derived, axis=1, join_axes=[df.index])
    # 更新data信息 如果没有输入new_dataframe_name 则不做操作
    if new_dataframe_name!='':
        update_df_in_HDFStore_by_name(store, new_dataframe_name, df) 

@app.route('/dc-feature-engineering/one-to-one', methods=['GET'])
def dc_feature_engineering_one_to_one():
    paras = {}
    try:
        # 获取ajax回传参数 
        store = pd.HDFStore(HDF5_path)
        df_name = request.args.get('df_name', None)
        assert HDF5_PREF+df_name+DATA_SUFF in store.keys(), \
            "dataframe %s not in store %s"%(HDF5_PREF+df_name+DATA_SUFF, store.filename)
        new_dataframe_name = request.args.get('new_dataframe_name', None)
        input_col = request.args.get('input_col', None)
        lambda_function_type = request.args.get('lambda_function_type', None)
        output_col = request.args.get('output_col', None)
        df = store[df_name+DATA_SUFF]
        ix.tag_meta_auto(df)
        if lambda_function_type=='isnull':
            new_dataframe = ix.derive_one_to_one(df, input_col, lambda a: not pd.isnull(a), output_col)
        store.put(new_dataframe_name+DATA_SUFF, new_dataframe)
        store.close()
        return redirect(url_for('dc_feature_engineering'))
    except Exception,e:
        return render_template('dc_error.html', e_message=e)
if __name__ == '__main__':
    app.run()
