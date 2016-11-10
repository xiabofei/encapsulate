#encoding=utf8
"""
encapsulate cch api as web service via flask
"""
from flask import Flask, request, redirect, url_for, session
import sys
# 检查参数
if len(sys.argv)<2: raise ValueError("input argv error")
# 加载需要的package
if sys.argv[1] == 's':
    another_cch_path = '/Users/xiabofei/Documents/cchdir'
    sys.path.append(another_cch_path)
    import src as ix
else:
    import cch as ix
import pandas as pd
import numpy as np
from os.path import join as pjoin
P = lambda p: pjoin(base_dir, p)
# 基础数据路径
base_dir = '/Users/xiabofei/Documents/cchdir/notebook/data'
# tmp数据路径
tmp_dir = '/Users/xiabofei/Documents/cchdir/encapsulate/data/'
# 表单处理
from flask_wtf import Form
from wtforms import StringField, SubmitField
from wtforms.validators import Required
# boostrap样式
from flask_bootstrap import Bootstrap
# 模板渲染
from flask import render_template
# 本地化时间
from flask_moment import Moment
# ipdb
from ipdb import set_trace as st


app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap = Bootstrap(app)
moment = Moment(app)

class NameForm(Form):
    file_name = StringField(u'csv文件名称', validators=[Required()]) 
    submit = SubmitField('Submit')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

@app.route('/dataset-register', methods=['GET', 'POST'])
def dataset_register():
    file_name = None
    form = NameForm()
    if request.method=='POST' and form.validate():
        paras = {}
        paras['file_name'] = form.file_name.data
        ret = similarity(paras)
        return render_template('similarity.html', grouping_rule = ret)
    return render_template('dataset_register.html', form = form, file_name = file_name)

def data_curation(path):
    """
    读入数据 转换为csv格式
    """
    df_SynPUFs = ix.read_csv_file_to_df(P(path), sep=',')
    df_SynPUFs = df_SynPUFs.drop(['Unnamed: 0'], axis=1)
    data_x_raw = df_SynPUFs[df_SynPUFs.columns.difference(['_LBL'])]
    data_y = df_SynPUFs['_LBL']
    # 中间数据本地序列化 (后期可以用数据库去存取)
    df_SynPUFs.to_pickle(tmp_dir+'df_SynPUFs')
    data_x_raw.to_pickle(tmp_dir+'data_x_raw')
    data_y.to_pickle(tmp_dir+'data_y')
    # 将本地序列化数据path存在session中
    session['pd_dataFrame'] = ['df_SynPUFs', 'data_x_raw', 'data_y']

def similarity(paras):
    """
    The SynPUFs-Similarity Process
    """
    data_curation(paras['file_name'])
    for k in session['pd_dataFrame']: globals()[k] = pd.read_pickle(tmp_dir+k)
    info_gain_index, info_gain_feature = ix.feature_selection('filter_method', 'information_gain', data_x_raw, data_y, 'k', 20)
    slct_cols = data_x_raw.columns[info_gain_index]
    data_x = data_x_raw[slct_cols]
    tf = ix.transform_data_with_learning_metric(metric='PCA', data_X=data_x, data_Y=data_y, params={'n_components':12})
    labels = ix.execute_clustering(data_X=tf,
            algorithm='KMeans',
            params={'init':'k-means++', 'n_init':10, 'n_clusters':2})
    sil_score = ix.evaluate_clustering_without_groundtruth(data_X=data_x, 
            cluster_id=labels.ix[:,0].values,
            metric='silhouette')
    pheno_dic, feature_cat = ix.phenotyping(data_X=data_x.astype(int),
            cluster_id=labels,
            max_cat_number = 10)
    bifeature = feature_cat.loc[feature_cat['feature_type'] == 'binary'].index
    data = pd.concat([data_y, data_x[bifeature], labels], axis=1)
    r1 = data['cluster_id'].value_counts()
    r2 = data.groupby('cluster_id').agg(np.count_nonzero)
    r3 = pd.concat([r1, r2], axis=1)
    r3.rename(columns={'cluster_id':'TOTAL', 'drv_lbl':'POS_OUTCOME'}, inplace=True)
    stat = ix.cluster_statistics(data_X=data_x, cluster_id=labels, f_type=feature_cat)
    tree, treegraph, accuracy = ix.grouping_rule_mining(data_x, labels, algorithm='CART', \
            params={'criterion':'gini', 'min_samples_leaf':10, 'max_depth':10})
    png = treegraph.create_png()
    rule_dict = ix.get_rules(tree.tree_, data_x.columns)
    ret = []
    for k, v in rule_dict.items():
        ret.append(u'患者群'+str(k))
        for path in v:
            ret.append(str(path))
    # return map(lambda r:"<p>"+str(r)+"</p>", ret)
    return ret

if __name__ == '__main__':
    app.run(debug=True)
