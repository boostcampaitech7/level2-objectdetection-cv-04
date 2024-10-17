import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

st.set_page_config(page_title="Load & Save Log", page_icon="🔍", layout="wide")

# Google Sheets API 설정
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('../streamlit/key_streamlit/nbc-7-project-2ee86f1b06b2.json', scope)
client = gspread.authorize(creds)

# 스프레드시트 열기 (URL 또는 스프레드시트 ID로 지정)
sheet = client.open_by_url('https://docs.google.com/spreadsheets/d/1Dre5HGZEvnyhmMDjIxTU_AtzM_A2Fm6wBpROrXat7qw/edit?gid=0#gid=0').sheet1  # or use .open('Sheet Name')

def load_data():
    data = sheet.get_all_values()
    headers = data.pop(0)
    df = pd.DataFrame(data, columns=headers)
    return df

def write_data(data):
    sheet.clear()
    sheet.update([data.columns.values.tolist()] + data.values.tolist())

# Streamlit 앱
def main():
    st.title('Google Sheets Data 관리 페이지')

    # 스프레드시트에서 데이터 가져오기
    data = sheet.get_all_values()
    
    if data: #
        # 첫 번째 행을 컬럼 이름으로 사용
        columns = data[0]
        
        # 나머지 행을 데이터로 사용
        rows = data[1:]
        
        # 컬럼 이름 표시
        st.write("Columns in the spreadsheet:")
        st.write(columns)
        
        # 데이터를 표로 표시
        st.write("Data from the spreadsheet:")
        st.table(rows)
    else:
        st.write("No data found in the spreadsheet.")
        # 데이터 로드

    if st.button('Load Data'):
        df = load_data()
        st.write("Loaded Data:")
        st.dataframe(df)

    # 데이터 쓰기
    st.subheader("Add New Row")
    new_row = {}
    for col in load_data().columns:
        new_row[col] = st.text_input(f"Enter {col}")

    if st.button('Add Row'):
        df = load_data()
        # df = df.append(new_row, ignore_index=True)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        write_data(df)
        st.success("새로운 행이 성공적으로 추가되었습니다!")

    # 데이터 수정
    st.subheader("Edit Data")
    df = load_data()
    edited_df = st.data_editor(df)

    if st.button('Save Changes'):
        write_data(edited_df)
        st.success("Changes saved successfully!")
        
if __name__ == '__main__':
    main()