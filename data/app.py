#thêm thư viện
import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os
from fuzzywuzzy import process
# Đường dẫn ảnh và dữ liệu
DATA_PATH = "D:/NCKH/building/data/nguoimiennuichat.csv"
IMAGE_PATH = "D:/NCKH/building/Picture1.jpg"

# Tải dataset
@st.cache_data
def load_data(data):
    try:
        df = pd.read_csv(data)
        return df
    except FileNotFoundError:
        st.error("File lỗi mất tiêu.")
        return None

def vectorize_text_to_cosine_mat(data):
    count_vect = CountVectorizer()
    cv_mat = count_vect.fit_transform(data)
    cosine_sim_mat = cosine_similarity(cv_mat)
    return cosine_sim_mat

@st.cache_data
def get_recommendation(title, cosine_sim_mat, df, num_of_rec=5):
    course_indices = pd.Series(df.index, index=df['course_title']).drop_duplicates()
    idx = course_indices[title]
    sim_scores = list(enumerate(cosine_sim_mat[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    selected_course_indices = [i[0] for i in sim_scores[:num_of_rec]]
    selected_course_scores = [i[1] for i in sim_scores[:num_of_rec]]
    result_df = df.iloc[selected_course_indices].copy()
    result_df['number_of_hits'] = selected_course_scores
    final_recommend_courses = result_df[['course_title', 'number_of_hits','subject_area', 'year', 'author']]
    return final_recommend_courses.head(num_of_rec)

@st.cache_data
def search_term_if_not_found(term, df, column='course_title', threshold=60):
    # Lọc danh sách các tên tài liệu
    course_titles = df[column].dropna().tolist()
    
    # Sử dụng fuzzy matching để tìm tài liệu gần đúng
    matches = process.extract(term, course_titles, limit=10)
    
    # Lọc ra các kết quả có độ tương đồng lớn hơn ngưỡng (threshold)
    filtered_matches = [match[0] for match in matches if match[1] >= threshold]
    
    # Trả về các hàng tương ứng trong DataFrame
    if filtered_matches:
        return df[df[column].isin(filtered_matches)]
    else:
        return pd.DataFrame()  # Không tìm thấy kết quả

# Hàm gọi API từ YEScale

def get_chatbot_response(messages):
    api_key = os.getenv('YESCALE_API_KEY')  # Sử dụng biến môi trường để lưu trữ khóa API
    try:
        response = requests.post(
            'https://api.yescale.io/v1/chat/completions',
            headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'},
            json={
                'max_tokens': 800,
                'messages': messages,
                'model': 'gpt-4-turbo',
                'temperature': 1,
                'top_p': 1
            }
        )
        
        if response.status_code == 200:
            return response.json().get('choices', [{}])[0].get('message', {}).get('content', 'Anh không nhận được tín hiệu.')
        elif response.status_code == 503:
            st.warning("Anh hơi mệt, thử lại sau nhé!")
            return "Anh hơi mệt, thử lại sau nhé!"
        else:
            st.error(f"API không nghe máy vì: {response.status_code} - {response.text}")
            return "API không nghe máy"
    except requests.exceptions.RequestException as e:
        st.error(f"API không nghe máy vì: {e}")
        return "API không nghe máy"
suggested_questions = [
    "Tài liệu nào phù hợp cho người mới bắt đầu học Python?",
    "Có tài liệu nào về học máy (machine learning) không?",
    "Làm thế nào để cài đặt thư viện scikit-learn?",
    "Có khóa học nào về trí tuệ nhân tạo không?",
    "Tài liệu nào tốt nhất để học lập trình web?"
]
# Giao diện ứng dụng
def main():
    st.image(IMAGE_PATH, caption="VMUva ^^", width=150)
    st.title('VMU Tư Vấn Tài Liệu')
    st.sidebar.image(IMAGE_PATH, use_container_width=True)
    menu = ('Trang chủ', 'Đề xuất tài liệu', 'VMUBot')
    choices = st.sidebar.selectbox('Menu', menu)
    df = load_data(DATA_PATH)
    
    if df is not None:
        if choices == 'Trang chủ':
            st.subheader('Trang chủ')
            st.write("Xem qua vài tài liệu nhé:")
            st.dataframe(df.head(10))
        
        elif choices == 'Đề xuất tài liệu':
            st.subheader('Đề xuất tài liệu')
            if 'course_title' in df.columns:
                try:
                    cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'].dropna())
                    search_term = st.text_input('Tìm gì nói đi bạn')
                    num_of_rec = st.sidebar.number_input('Số lượng đề xuất', 1, 30, 3)
                    
                    if st.button('Đề xuất'):
                        if search_term and search_term.strip():
                            #Tim kiem gan dung
                            results = search_term_if_not_found(search_term, df)
                            if not results.empty:
                                st.write("Các tài liệu đề xuất:")
                                for index, row in results.iterrows():
                                    stc.html(f"""
                                        <div style="padding:10px;margin:5px;border-radius:5px;background-color:#FFFFCC;">
                                            <h4>{row['course_title']}</h4>
                                            <p><b>Subject Area:</b> {row['subject_area']}</p>
                                            <p><b>Year:</b> {row['year']}</p>
                                            <p><b>Author:</b> {row['author']}</p>
                                            <p><b>Similarity Score:</b> {row['number_of_hits']:.2f}</p>
                                            <a href="https://drive.google.com/drive/folders/1J1AHkVgBuoFMm8rh7i1HLLDIrNveLP7y?usp=sharing">Link tai lieu</a>
                                        </div>
                                    """, height=300)
                            else:
                                st.warning("Chịu rồi không thấy@@")
                        else:
                            st.warning("Vui lòng nhập từ khóa tìm kiếm hợp lệ.")
                except Exception as e:
                    st.error(f"Hình như không có: {e}")
            else:
                st.error("Không thấy cái nào là 'course_title' cả")
        
        elif choices == 'VMUBot':
            st.subheader('VMUBot')
            st.write("VMUBot sẵn sàng hỗ trợ bạn!")
            
            st.title('Hỏi gì hỏi đi bạn ^^')



            st.write("Bạn có thể hỏi những câu hỏi sau:")
            for question in suggested_questions:
                if st.button(question):
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    response = get_chatbot_response(st.session_state.chat_history)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []

            user_input = st.text_input('Tôi: ', '')

            if user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                response = get_chatbot_response(st.session_state.chat_history)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
            for chat in st.session_state.chat_history:
                if chat["role"] == "user":
                    st.write(f"**Tôi:** {chat['content']}")
                else:
                    st.write(f"**VMUbot:** {chat['content']}")
#Chạy ứng dụng
if __name__ == '__main__':
    main()