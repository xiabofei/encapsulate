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
from wtforms import StringField, SubmitField, SelectField
from wtforms.validators import DataRequired
# boostrap样式
from flask_bootstrap import Bootstrap
# 模板渲染
from flask import render_template
# 本地化时间
from flask_moment import Moment
# ipdb
from ipdb import set_trace as st
# json
import json
from pandas.io.json import json_normalize


# 起一个app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap = Bootstrap(app)
moment = Moment(app)

@app.route('/')
def index():
    return render_template('default.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

# 数据获取
class RegForm(Form):
    file_name = StringField(u'csv文件名称', default=u'SynPUFs.curated_1.csv', validators=[DataRequired()]) 
    submit = SubmitField('Submit')
@app.route('/dataset-register', methods=['GET', 'POST'])
def dataset_register():
    """
    读入数据 & 数据分割 & 存中间数据
    """
    paras = {}
    paras['file_name'] = None
    form = RegForm()
    paras['form'] = form
    if request.method=='POST' and form.validate():
        paras['file_name'] = form.file_name.data
        try:
            # 读入数据
            df_SynPUFs = ix.read_csv_file_to_df(P(paras['file_name']), sep=',')
            # 数据按列分割
            df_SynPUFs = df_SynPUFs.drop(['Unnamed: 0'], axis=1)
            data_x_raw = df_SynPUFs[df_SynPUFs.columns.difference(['_LBL'])]
            data_y = df_SynPUFs['_LBL']
            # 存中间数据 (后期用数据库存取)
            df_SynPUFs.to_pickle(tmp_dir+'df_SynPUFs')
            data_x_raw.to_pickle(tmp_dir+'data_x_raw')
            data_y.to_pickle(tmp_dir+'data_y')
            # session中存中间数据
            session['pd_dataFrame'] = ['df_SynPUFs', 'data_x_raw', 'data_y']
            paras['pd_shape'] = [df_SynPUFs.shape, data_x_raw.shape, data_y.shape]
            paras['curated_data'] = session['pd_dataFrame']
            return render_template('dataset_register.html', **paras)
        except Exception,e:
            return render_template('error.html', e_message=e)
    return render_template('dataset_register.html', **paras)

# 特征选择
class FeaSelForm(Form):
    strategy = SelectField(u'特征选择策略', choices=[("filter_method","filter_method")]) 
    feature_selection_method = SelectField(u'特征选择方法', choices=[('information_gain','information_gain')])
    data_X = StringField(u'样本数据', default='data_x_raw', validators=[DataRequired()]) 
    data_Y = StringField(u'样本标签', default='data_y', validators=[DataRequired()])
    selection_parameter = SelectField(u'选择指标度量', choices=[("k","k"), ("percentile","percentile"), ("threshold","threshold")])
    selection_parameter_value = StringField(u'最大选择特征数', default='20', validators=[DataRequired()])
    submit = SubmitField('Submit')
@app.route('/feature-selection', methods=['GET', 'POST'])
def feature_selection():
    """
    特征选择
    """
    paras = {}
    form = FeaSelForm()
    paras['form'] = form
    if request.method=='POST' and form.validate():
        try:
            # 模拟从数据库获取数据
            data_x_raw = pd.read_pickle(tmp_dir+form.data_X.data)
            data_y = pd.read_pickle(tmp_dir+form.data_Y.data)
            # 选择相关度最高的k个feature
            info_gain_index, info_gain_feature = ix.feature_selection(
                    str(form.strategy.data),
                    str(form.feature_selection_method.data),
                    data_x_raw,
                    data_y,
                    str(form.selection_parameter.data),
                    int(form.selection_parameter_value.data))
            slct_cols = data_x_raw.columns[info_gain_index]
            # 模拟向数据库写入selected的数据
            data_x = data_x_raw[slct_cols]
            data_x.to_pickle(tmp_dir+'data_x') 
            # head_sample = data_x.head().to_json(force_ascii=False)
            head_sample = data_x.head(n=20).to_html(classes='table table-striped')
            paras['head_sample'] = head_sample
            paras['selected_feature'] = info_gain_feature
            print type(info_gain_feature)
            return render_template('feature_selection.html', **paras)
        except Exception,e:
            return render_template('error.html', e_message=e)
    return render_template('feature_selection.html', **paras)

def similarity(paras):
    """
    The SynPUFs-Similarity Process
    """
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
    return ret

if __name__ == '__main__':
    app.run(debug=True)
