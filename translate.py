import pandas as pd
from googletrans import Translator


def translate_text_in_excel(input_file_path, output_file_path):
    # 读取文件
    excel_file = pd.ExcelFile(input_file_path)
    # 获取指定工作表中的数据，这里假设只有一个工作表'Sheet1'，若有多个需按需修改
    df = excel_file.parse('Sheet1')
    # 初始化翻译器
    translator = Translator()
    # 对 text 列进行翻译，并将结果存储在 翻译 列中，如果 翻译 列存在非空值则跳过
    def translate_row(row):
        if pd.notnull(row['翻译']):
            return row['翻译']
        return translator.translate(row['text'], dest='zh-CN').text

    df['翻译'] = df.apply(translate_row, axis=1)

    # 将结果保存为新的 Excel 文件
    df.to_excel(output_file_path, index=False)


if __name__ == "__main__":
    input_file_path = r'data\Event2012\open_set\20\20_all_clustered_data.xlsx'
    output_file_path = r'data\Event2012\open_set\20\20_all_clustered_data_translate.xlsx'
    translate_text_in_excel(input_file_path, output_file_path)