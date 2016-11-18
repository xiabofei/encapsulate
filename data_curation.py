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
    if request.method=='POST' and form.validate():
        try:
            return render_template('dc_data_exploration.html', **paras)
        except Exception,e:
            return render_template('dc_error.html', e_message=e)
    return render_template('dc_data_exploration.html', **paras)
if __name__ == '__main__':
    app.run()
