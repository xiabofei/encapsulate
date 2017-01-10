#coding=utf8
import selenium.webdriver as webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.touch_actions import TouchActions 
import time
import os

from ipdb import set_trace as st

def wait_for(waiting):
    time.sleep(waiting)

def open_datacuration_page(driver, seed_url):
    driver.get(seed_url)

def auto_data_preview(driver):
    wait = WebDriverWait(driver, 30)
    # 1. 起一个action实例
    actions = ActionChains(driver)
    # 2. 点击'数据预览'
    wait.until(EC.visibility_of_element_located((By.LINK_TEXT, '数据预览')))
    data_presee = driver.find_element_by_link_text('数据预览')
    actions.move_to_element(data_presee).perform()
    wait_for(1)
    data_presee.click()
    # 3. 输入'csv文件路径'和'读入行数' 并点击提交
    wait.until(EC.visibility_of_element_located((By.ID, 'file_list')))
    wait.until(EC.visibility_of_element_located((By.ID, 'nrows_list')))
    wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="csv-sumbit"]')))
    file_list = driver.find_element_by_id('file_list')
    nrows_list = driver.find_element_by_id('nrows_list')
    csv_submit = driver.find_element_by_xpath('//*[@id="csv-sumbit"]')
    wait_for(1)
    file_list.clear()
    file_list.send_keys('heart.csv')
    wait_for(1)
    nrows_list.clear()
    nrows_list.send_keys('1000')
    wait_for(1)
    csv_submit.click()
    # 4. 点开df_heart 并展现读入的数据 再关闭
    wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="accordion"]/div/div[1]/a')))
    df_heart_view = driver.find_element_by_xpath('//*[@id="accordion"]/div/div[1]/a')
    actions = ActionChains(driver)
    actions.move_to_element(df_heart_view).perform()
    wait_for(1)
    df_heart_view.click()
    touch_actions = TouchActions(driver)
    touch_actions.scroll_from_element(df_heart_view, 0, 400).perform()
    wait_for(1)
    df_heart_view.click()
    # 5. 展示已经读入HDFStore的文件
    wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="df_sumbit"]')))
    wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="store_path"]')))
    df_submit = driver.find_element_by_xpath('//*[@id="df_sumbit"]')
    store_path = driver.find_element_by_xpath('//*[@id="store_path"]')
    store_path.send_keys('data_curation.h5')
    wait_for(1)
    df_submit.click()
    wait_for(1)
    wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="accordion"]/div/div[1]/a')))
    df_show = driver.find_element_by_xpath('//*[@id="accordion"]/div/div[1]/a')
    df_show.click()
    touch_actions = TouchActions(driver)
    touch_actions.scroll_from_element(df_show, 0, 800).perform()
    wait_for(1)
    df_show.click()
    wait_for(2)

def auto_data_exploration(driver):
    wait = WebDriverWait(driver, 30)
    # 1. 进入数据探索界面
    entry_li = driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/ul/li[2]/a')
    entry_li.click()
    wait_for(1)
    # 2. 进行数据探索
    wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="exploration"]')))
    data_exploration_bt = driver.find_element_by_xpath('//*[@id="exploration"]')
    data_exploration_bt.click()
    wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="pagecontent"]/ul/li/a')))
    df_heart_tab = driver.find_element_by_xpath('//*[@id="pagecontent"]/ul/li/a')
    touch_actions = TouchActions(driver)
    touch_actions.scroll(0, 800).perform()
    touch_actions = TouchActions(driver)
    touch_actions.scroll(0, -400).perform()
    # 3. 展示每个特征的详细分布比例 
    xpaths = [
            '//*[@id="menudf_heart"]/table/tbody/tr[1]/td[6]/a',
            '//*[@id="menudf_heart"]/table/tbody/tr[3]/td[6]/a',
            '//*[@id="menudf_heart"]/table/tbody/tr[5]/td[6]/a'
            ]
    show_feature_detail(driver, wait, xpaths)

def show_feature_detail(driver, wait, xpaths):
    for xpath in xpaths:
        wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
        feature_detail = driver.find_element_by_xpath(xpath)
        feature_detail.click()
        wait_for(2)
        feature_detail.click()
        wait_for(1)

def auto_feature_type_convert(driver):
    wait = WebDriverWait(driver, 30)
    # 1. 进入特征工程界面
    entry_li = driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/ul/li[4]/a')
    entry_li.click()
    wait_for(1)
    # 2. 特征类型转换
    wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="feature_convert"]/div/div/div[1]/h4/a')))
    df_heart_convert = driver.find_element_by_xpath('//*[@id="feature_convert"]/div/div/div[1]/h4/a')
    df_heart_convert.click()
    wait_for(2)
    touch_actions = TouchActions(driver)
    touch_actions.scroll(0, 200).perform()
    feature_type_xpaths = [
                    (
                        '//*[@id="collapseupdatedf_heart"]/div/table/tbody/tr[3]/td[1]/input',
                        '//*[@id="collapseupdatedf_heart"]/div/table/tbody/tr[3]/td[5]/select/option[6]'
                    ),
                    (
                        '//*[@id="collapseupdatedf_heart"]/div/table/tbody/tr[7]/td[1]/input',
                        '//*[@id="collapseupdatedf_heart"]/div/table/tbody/tr[7]/td[5]/select/option[6]'
                    ),
                    (
                        '//*[@id="collapseupdatedf_heart"]/div/table/tbody/tr[11]/td[1]/input',
                        '//*[@id="collapseupdatedf_heart"]/div/table/tbody/tr[11]/td[5]/select/option[6]'
                    ),
                    (
                        '//*[@id="collapseupdatedf_heart"]/div/table/tbody/tr[12]/td[1]/input',
                        '//*[@id="collapseupdatedf_heart"]/div/table/tbody/tr[12]/td[5]/select/option[6]'
                    ),
                    (
                        '//*[@id="collapseupdatedf_heart"]/div/table/tbody/tr[13]/td[1]/input',
                        '//*[@id="collapseupdatedf_heart"]/div/table/tbody/tr[13]/td[5]/select/option[6]'
                    )
                ]
    _convert_feature_type(driver, wait, feature_type_xpaths)
    trans_bt = driver.find_element_by_xpath('//*[@id="coltypeupdatedf_heart"]')
    trans_bt.click()
    wait_for(5)
    # 3. 回到数据摸底页面 查看类型转换后的结果
    auto_data_exploration(driver)
    entry_li = driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/ul/li[4]/a')
    entry_li.click()
    wait_for(1)

def _convert_feature_type(driver, wait, xpaths):
    for xpath in xpaths: 
        wait.until(EC.visibility_of_element_located((By.XPATH, xpath[0])))
        feature_check = driver.find_element_by_xpath(xpath[0])
        feature_check.click()
        wait.until(EC.visibility_of_element_located((By.XPATH, xpath[1])))
        select_type = driver.find_element_by_xpath(xpath[1]) 
        select_type.click()

def auto_factor_feature_binning(driver):
    wait = WebDriverWait(driver, 30)
    entry_li = driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/ul/li[4]/a')
    entry_li.click()
    wait_for(1)
    # 1. 执行factor类型的特征打散
    wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="meta_factor"]/div/div/div[1]/h4/a')))
    df_heart = driver.find_element_by_xpath('//*[@id="meta_factor"]/div/div/div[1]/h4/a')
    df_heart.click()
    wait_for(1)
    wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="collapseprodf_heart"]/div/input[2]')))
    new_dataframe_name = driver.find_element_by_xpath('//*[@id="collapseprodf_heart"]/div/input[2]')
    new_dataframe_name.clear()
    new_dataframe_name.send_keys('df_heart_derived_factor')
    wait_for(1)
    feature_indexs = [1,2,3,4,5,6,7,8]
    for index in feature_indexs:
        xpath = '//*[@id="collapseprodf_heart"]/div/table/tbody/tr['+str(index)+']/td[1]/input'
        check_box = driver.find_element_by_xpath(xpath)
        check_box.click()
        wait_for(0.5)
    # st(context=21)
    factor_binning_bt = driver.find_element_by_xpath('//*[@id="featureengineerdf_heart"]') 
    factor_binning_bt.click()
    wait_for(1)
    # 2. 查看打散后的生成的dataframe
    wait.until(EC.visibility_of_element_located((By.LINK_TEXT, '数据预览')))
    data_presee = driver.find_element_by_link_text('数据预览')
    data_presee.click()
    wait_for(1)
    store_path = driver.find_element_by_xpath('//*[@id="store_path"]')
    store_path.clear()
    store_path.send_keys('data_curation.h5')
    wait_for(1)
    df_submit = driver.find_element_by_xpath('//*[@id="df_sumbit"]')
    df_submit.click()
    wait_for(2)
    df_heart_derived_factor_show = driver.find_element_by_xpath('//*[@id="accordion"]/div[3]/div[1]/a')
    df_heart_derived_factor_show.click()
    touch_actions = TouchActions(driver)
    touch_actions.scroll(0, 800).perform()
    wait_for(2)
    df_heart_derived_factor_show.click()
    wait_for(2)
    # 3. 回到feature engineering页面 删除原有的factor feature
    entry_li = driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/ul/li[3]/a')
    entry_li.click()
    wait_for(2)
    wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="accordion"]/form/div/div[1]/h4/a')))
    df_heart = driver.find_element_by_xpath('//*[@id="accordion"]/form/div/div[1]/h4/a') 
    df_heart.click()
    wait_for(2)
    feature_indexs = [2,3,6,7,9,11,12,13]
    for index in feature_indexs:
        xpath = '//*[@id="collapsedf_heart"]/div/table/tbody/tr['+str(index)+']/td[1]/input'
        check_box = driver.find_element_by_xpath(xpath)
        check_box.click()
        wait_for(0.5)
    filter_bt = driver.find_element_by_xpath('//*[@id="collapsedf_heart"]/div/input')
    filter_bt.click()
    wait_for(2)
    # 4. 重新执行data exploration
    entry_li = driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/ul/li[2]/a')
    entry_li.click()
    wait_for(1)
    wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="exploration"]')))
    data_exploration_bt = driver.find_element_by_xpath('//*[@id="exploration"]')
    data_exploration_bt.click()
    wait_for(1)
    wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="pagecontent"]/ul/li[1]/a')))
    touch_actions = TouchActions(driver)
    touch_actions.scroll(0, 800).perform()
    wait_for(2) 

def auto_concat_dataframe(driver):
    wait = WebDriverWait(driver, 30)
    # 1. 回到'特征工程'界面
    entry_li = driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/ul/li[4]/a')
    entry_li.click()
    wait_for(1)
    touch_actions = TouchActions(driver)
    touch_actions.scroll(0, 800).perform()
    wait_for(2) 
    # 2. 合并的dataframe 生成dataframe
    left_dataframe = driver.find_element_by_xpath('//*[@id="concat_dataframe_action"]/div/div[2]/div[1]/div/div[2]/select[1]/option[2]')
    left_dataframe.click()
    wait_for(1)
    left_dataframe = driver.find_element_by_xpath('//*[@id="concat_dataframe_action"]/div/div[2]/div[1]/div/div[2]/select[1]/option[1]')
    left_dataframe.click()
    wait_for(1)
    right_dataframe = driver.find_element_by_xpath('//*[@id="concat_dataframe_action"]/div/div[2]/div[1]/div/div[2]/select[2]/option[2]')
    right_dataframe.click()
    axis = driver.find_element_by_xpath('//*[@id="concat_dataframe_action"]/div/div[2]/div[1]/div/div[3]/select[1]/option[2]')
    axis.click()
    new_dataframe_name = driver.find_element_by_xpath('//*[@id="concat_dataframe_action"]/div/div[2]/div[1]/div/div[3]/input[2]')
    new_dataframe_name.clear()
    wait_for(1)
    new_dataframe_name.send_keys('df_heart_output')
    wait_for(1)
    st(context=21)
    concat_submit = driver.find_element_by_xpath('//*[@id="concat_dataframe"]')
    concat_submit.click()
    wait_for(2)
    # 3. 回到'数据预览'界面查看新生成的dataframe
    entry_li = driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/ul/li[1]/a')
    entry_li.click()
    wait_for(1)
    wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="df_sumbit"]')))
    df_submit = driver.find_element_by_xpath('//*[@id="df_sumbit"]')
    df_submit.click()
    wait_for(1)
    wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="accordion"]/div[5]/div[1]/a')))
    df_heart_output = driver.find_element_by_xpath('//*[@id="accordion"]/div[5]/div[1]/a')
    df_heart_output.click()
    touch_actions = TouchActions(driver)
    touch_actions.scroll(0, 600).perform()
    wait_for(2) 
    # 4. 生成csv文件
    create_csv = driver.find_element_by_xpath('//*[@id="storecsvdf_heart_output_DATA"]')
    create_csv.click()
    wait_for(2) 


def open_similarity_page(driver, seed_url):
    driver.get(seed_url)

def similarity_process(driver):
    wait = WebDriverWait(driver, 30)
    wait_for(2)
    # 1. 数据获取
    wait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[1]/div/div[2]/ul/li[1]/a')))
    data_register = driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/ul/li[1]/a')
    data_register.click()
    wait_for(2)
    wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="file_name"]')))
    wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="label_name"]')))
    file_name = driver.find_element_by_xpath('//*[@id="file_name"]')
    label_name = driver.find_element_by_xpath('//*[@id="label_name"]')
    file_name.clear()
    file_name.send_keys('df_heart_output_DATA.csv')
    wait_for(1)
    label_name.clear()
    label_name.send_keys('label')
    wait_for(1)
    submit_bt = driver.find_element_by_xpath('//*[@id="submit"]')
    submit_bt.click()
    wait_for(2)
    # 2. 特征选择
    feature_selection = driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/ul/li[2]/a')
    feature_selection.click()
    wait_for(2)
    max_select_feature = driver.find_element_by_xpath('//*[@id="selection_parameter_value"]')
    max_select_feature.clear()
    wait_for(1)
    max_select_feature.send_keys('25')
    wait_for(1)
    submit_bt = driver.find_element_by_xpath('//*[@id="submit"]')
    submit_bt.click()
    touch_actions = TouchActions(driver)
    touch_actions.scroll(0, 2000).perform()
    wait_for(2) 
    touch_actions = TouchActions(driver)
    touch_actions.scroll(0, -1000).perform()
    wait_for(2) 
    touch_actions = TouchActions(driver)
    touch_actions.scroll(0, -500).perform()
    wait_for(2) 
    # 3. 患者聚类
    patient_cluster = driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/ul/li[3]/a')
    patient_cluster.click()
    wait_for(2)
    touch_actions = TouchActions(driver)
    touch_actions.scroll(0, 400).perform()
    wait_for(2) 
    touch_actions = TouchActions(driver)
    touch_actions.scroll(0, -200).perform()
    wait_for(2) 
    feature_reduce_dimension = driver.find_element_by_xpath('//*[@id="selection_parameter_value"]')
    feature_reduce_dimension.clear()
    wait_for(2) 
    feature_reduce_dimension.send_keys('5')
    cluster_number = driver.find_element_by_xpath('//*[@id="n_clusters"]')
    cluster_number.clear()
    wait_for(2) 
    cluster_number.send_keys('2')
    wait_for(2) 
    submit_bt = driver.find_element_by_xpath('//*[@id="submit"]')
    submit_bt.click()
    wait_for(2) 
    touch_actions = TouchActions(driver)
    touch_actions.scroll(0, 2000).perform()
    wait_for(2) 
    touch_actions = TouchActions(driver)
    touch_actions.scroll(0, -1000).perform()
    wait_for(2) 
    touch_actions = TouchActions(driver)
    touch_actions.scroll(0, -500).perform()
    wait_for(2) 
    # 4. 规则挖掘
    rule_mining = driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/ul/li[4]/a')
    rule_mining.click()
    wait_for(2)
    label = driver.find_element_by_xpath('//*[@id="labels"]')
    label.clear()
    wait_for(2)
    label.send_keys('labels')
    wait_for(2)
    submit_bt = driver.find_element_by_xpath('//*[@id="submit"]')
    submit_bt.click()
    wait_for(2)
    touch_actions = TouchActions(driver)
    touch_actions.scroll(0, 2000).perform()
    wait_for(2) 

if __name__ == '__main__':
    try:
        driver = webdriver.Chrome('./chromedriver')
        driver.set_page_load_timeout(30)
        # data curation
        open_datacuration_page(driver, 'http://127.0.0.1:5000')
        auto_data_preview(driver)
        auto_data_exploration(driver)
        auto_feature_type_convert(driver)
        auto_factor_feature_binning(driver)
        auto_concat_dataframe(driver)
        # similarity patient clustering
        open_similarity_page(driver, 'http://127.0.0.1:5001')
        similarity_process(driver)
    finally:
        if driver is not None:
            driver.quit()
