from typing import Text, List, Dict, Any
from rasa_sdk import Action, Tracker
from rasa_sdk.events import UserUtteranceReverted
from rasa_sdk.executor import CollectingDispatcher
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import  pandas as pd
import numpy as np
import heapq
import json


class ActionRating(Action):
    def name(self) -> Text:
        return "action_rating"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        df = pd.read_csv(filepath_or_buffer='/Users/DELL/RASA/Hotel_Travel/Model_Raking/hotel_raking.csv')
        corpus = list(df["district"])
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        accommodation_slot = tracker.get_slot("accommodation")
        district_slot = tracker.get_slot("district")
        if district_slot is not None and accommodation_slot is not None:
            key_vec = vectorizer.transform([district_slot])
            similarity_scores = cosine_similarity(X,key_vec)
            max_similarity_index = np.argmax(similarity_scores)
            keys = list(df["district"])
            point = keys[max_similarity_index]
            try:
                accommodation_rating = float(accommodation_slot)
                filtered_rows = df[(df["accommodation"] == accommodation_rating) & (df["district"] == point)]
                hotels = filtered_rows['name'].head(5).tolist()

                if len(hotels) > 0:
                    message = f"Tôi có thể gợi ý cho bạn vài khách sạn {accommodation_rating} sao tốt nhất mà tôi kiếm được ở khu vực {point} như:\n" + "\n".join(f"👉 {hotel}" for hotel in hotels)
                else:
                    message = f"Không tìm thấy khách sạn {accommodation_rating} sao ở khu vực {point}."

                dispatcher.utter_message(text=message)

            except ValueError:
                message = "Cú pháp không hợp lệ. Vui lòng nhập lại"
                dispatcher.utter_message(text=message)

        else:
            message = "Xin lỗi, tôi không tìm được khách sạn nào dựa trên yêu cầu của bạn"
            dispatcher.utter_message(text=message)
        return []


class ActionPrice(Action):
    def name(self) -> Text:
        return "action_price_hotel"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        df = pd.read_csv(filepath_or_buffer='/Users/DELL/RASA/Hotel_Travel/Model_Raking/hotel_raking.csv')
        corpus = list(df["district"])
        
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        
        average_price_slot = tracker.get_slot("average_price")
        district_slot = tracker.get_slot("district")
        
        if district_slot is not None and average_price_slot is not None:
    
            key_vec = vectorizer.transform([district_slot])
            similarity_scores = cosine_similarity(X, key_vec)
            max_similarity_index = np.argmax(similarity_scores)
            point = corpus[max_similarity_index]
            
            try:
                average_price = int(average_price_slot.replace('.', ''))
                filtered_rows = df[(df["average_price"] <= average_price) & (df["district"] == point)]
                hotels = filtered_rows['name'].head(5).tolist()
                
                if len(hotels) > 0:
                    message = f"Tôi có thể gợi ý cho bạn vài khách sạn với giá tầm {average_price_slot} đồng tốt nhất mà tôi kiếm được ở khu vực {point} như:\n" + "\n".join(f"👉 {hotel}" for hotel in hotels)
                else:
                    message = f"Không tìm thấy khách sạn với giá tầm {average_price_slot} đồng ở khu vực {point}."
                
                dispatcher.utter_message(text=message)
            
            except ValueError:
                message = "Cú pháp không hợp lệ. Vui lòng nhập lại."
                dispatcher.utter_message(text=message)
        
        else:
            message = "Xin lỗi, tôi không tìm được khách sạn nào dựa trên yêu cầu của bạn"
            dispatcher.utter_message(text=message)
        
        return []

class ActionAlloption(Action):
    def name(self) -> Text:
        return "action_all_option"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:     
        df = pd.read_csv(filepath_or_buffer='/Users/DELL/RASA/Hotel_Travel/Model_Raking/hotel_raking.csv')
        corpus = list(df["district"])
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        accommodation_slot = tracker.get_slot("accommodation")
        average_price_slot = tracker.get_slot("average_price")
        district_slot = tracker.get_slot("district")
        if district_slot is not None and average_price_slot is not None and accommodation_slot is not None:
            key_vec = vectorizer.transform([district_slot])
            similarity_scores = cosine_similarity(X,key_vec)
            max_similarity_index = np.argmax(similarity_scores)
            keys = list(df["district"])
            point = keys[max_similarity_index]
            try:
                accommodation_rating = float(accommodation_slot)
                average_price = int(average_price_slot.replace('.', ''))
                filtered_rows = df[(df["accommodation"] == accommodation_rating) & (df["average_price"] <= average_price) & (df["district"] == point)]
                hotels = filtered_rows['name'].head(5).tolist()
                if len(hotels) > 0:
                    message = f"Tôi có thể gợi ý cho bạn vài khách sạn {accommodation_rating} sao với giá tầm {average_price_slot} đồng tốt nhất mà tôi kiếm được ở khu vực {point} như:\n" + "\n".join(f"👉 {hotel}" for hotel in hotels)
                else:
                    message = f"Không tìm thấy khách sạn {accommodation_rating} sao với giá tầm {average_price_slot} đồng ở khu vực {point}."
                dispatcher.utter_message(text=message)

            except ValueError:
                message = "Cú pháp không hợp lệ. Vui lòng nhập lại"
                dispatcher.utter_message(text=message)

        else:
            message = "Xin lỗi, tôi không tìm được khách sạn nào dựa trên yêu cầu của bạn"
            dispatcher.utter_message(text=message)
        return []

class ActionAddress(Action):
    def name(self) -> Text:
        return "action_name_hotel"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        df = pd.read_csv(filepath_or_buffer='/Users/DELL/RASA/Hotel_Travel/Model_Raking/hotel_raking.csv')
        corpus = list(df["name"])
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)

        hotel_slot = tracker.get_slot("name_hotel")
        
        if hotel_slot is not None:
            key_vec = vectorizer.transform([hotel_slot])
            similarity_scores = cosine_similarity(X, key_vec)
            max_similarity_index = np.argmax(similarity_scores)
            max_similarity_score = similarity_scores[max_similarity_index]
            
            if max_similarity_score >= 0.7: 
                point = df.iloc[max_similarity_index]["name"]
                hotels = df[df["name"] == point]

                address = ", ".join(hotels["address"].tolist())
                highlight = ", ".join(hotels["highlight"].tolist())
                formatted_number = ", ".join('{:,.0f}'.format(price) for price in hotels["average_price"].tolist())
                price = formatted_number.replace(',', '.')
                amenity = ", ".join(str(amenity) for amenity in hotels["amenity"].tolist())
                link = ", ".join(str(link) for link in hotels["link"].tolist())

                dispatcher.utter_message(text=f"Dưới đây là một vài thông tin về {point}")
                message = "- Địa chỉ: {}\n" \
                            "- Điểm nổi bật: {}\n" \
                            "- Tiện ích: {}\n" \
                            "- Giá trung bình giao động từ: {} đồng\n" \
                            "- Bạn có thể tham khảo đặt phòng tại trang Expedia: {}\n"

                dispatcher.utter_message(text=message.format(address, highlight, amenity, price, link))
                dispatcher.utter_message(text="Bạn có muốn biết thêm thông tin về các địa điểm du lịch, nhà hàng, giao thông công cộng gần khách sạn không? Nếu có, hãy nhập theo cú pháp sau.",
                                            buttons=[
                                                {"title": "Địa điểm tham quan", "payload": "/question_location_nearby"},
                                                {"title": "Nhà hàng", "payload": "/question_restaurant_nearby"},
                                                {"title": "Giao thông công cộng", "payload": "/question_around_nearby"}
                                            ])
            else:
                dispatcher.utter_message(text="Xin lỗi, hiện tại khách sạn này không có trong hồ sơ của tôi")
        else:
            dispatcher.utter_message(text="Xin lỗi, vui lòng nhập lại")
        
        return []

class ActionLocationNearby(Action):
    def name(self) -> Text:
        return "action_location_nearby"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        df = pd.read_csv('/Users/DELL/RASA/Hotel_Travel/Model_Raking/hotel_raking.csv')
        df_location = pd.read_csv('/Users/DELL/RASA/Hotel_Travel/Clean/Expedia/Location_nearby_Hotel.csv')
        df_location_hotel = pd.merge(df, df_location, on='id', how='inner')
        corpus = list(df_location_hotel["name"])
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        hotel_slot = tracker.get_slot("name_hotel")
        if hotel_slot is not None:
            key_vec = vectorizer.transform([hotel_slot])
            similarity_scores = cosine_similarity(X, key_vec)

            if np.max(similarity_scores) < 0.7:
                dispatcher.utter_message(text="Không tìm thấy tên khách sạn phù hợp.")
            else:
                max_similarity_index = np.argmax(similarity_scores)
                hotel_name = corpus[max_similarity_index]
                hotels = df_location_hotel[(df_location_hotel["name"] == hotel_name)]
                message_1 = f"Các địa điểm gần {hotel_name} gồm:\n"
                dispatcher.utter_message(text=message_1)
                message_2 = ""
                for index, row in hotels.iterrows():
                    location = row['location_place']
                    distance = row['location_time']
                    message_2 += f"+ {location} - thời gian di chuyển: {distance}\n"
                dispatcher.utter_message(text=message_2)
        else:
            dispatcher.utter_message(text="Xin lỗi, hiện tại khách sạn này không có trong hồ sơ của tôi")
        return []
    
class ActionRestaurantNearby(Action):
    def name(self) -> Text:
        return "action_restaurant_nearby"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        df = pd.read_csv('/Users/DELL/RASA/Hotel_Travel/Model_Raking/hotel_raking.csv')
        df_restaurant = pd.read_csv('/Users/DELL/RASA/Hotel_Travel/Clean/Expedia/Restaurant_nearby_Hotel.csv')
        df_restaurant_hotel = pd.merge(df, df_restaurant, on='id', how='inner')
        corpus = list(df_restaurant_hotel["name"])
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        hotel_slot = tracker.get_slot("name_hotel")
        if hotel_slot is not None:
            key_vec = vectorizer.transform([hotel_slot])
            similarity_scores = cosine_similarity(X, key_vec)

            if np.max(similarity_scores) < 0.7:
                dispatcher.utter_message(text="Không tìm thấy tên khách sạn phù hợp.")
            else:
                max_similarity_index = np.argmax(similarity_scores)
                hotel_name = corpus[max_similarity_index]
                hotels = df_restaurant_hotel[(df_restaurant_hotel["name"] == hotel_name)]
                message_1 = f"Các nhà hàng gần {hotel_name} gồm:\n"
                dispatcher.utter_message(text=message_1)
                message_2 = ""
                for index, row in hotels.iterrows():
                    restaurant = row['restaurant_place']
                    distance = row['restaurant_time']
                    message_2 += f"+ {restaurant} - thời gian di chuyển: {distance}\n"
                dispatcher.utter_message(text=message_2)
        else:
            dispatcher.utter_message(text="Xin lỗi, hiện tại khách sạn này không có trong hồ sơ của tôi")
        return []

class ActionAroundNearby(Action):
    def name(self) -> Text:
        return "action_around_nearby"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        df = pd.read_csv('/Users/DELL/RASA/Hotel_Travel/Model_Raking/hotel_raking.csv')
        df_around = pd.read_csv('/Users/DELL/RASA/Hotel_Travel/Clean/Expedia/Around_nearby_Hotel.csv')
        df_around_hotel = pd.merge(df, df_around, on='id', how='inner')

        corpus = list(df_around_hotel["name"])
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        hotel_slot = tracker.get_slot("name_hotel")

        if hotel_slot is not None:
            key_vec = vectorizer.transform([hotel_slot])
            similarity_scores = cosine_similarity(X, key_vec)

            if np.max(similarity_scores) < 0.7:
                dispatcher.utter_message(text="Không tìm thấy tên khách sạn, vui lòng nhập lại")
            else:
                max_similarity_index = np.argmax(similarity_scores)
                hotel_name = corpus[max_similarity_index]
                hotels = df_around_hotel[(df_around_hotel["name"] == hotel_name)]
                message_1 = f"Các điểm giao thông công cộng gần {hotel_name} gồm:\n"
                dispatcher.utter_message(text=message_1)
                message_2 = ""
                for index, row in hotels.iterrows():
                    around = row['around_place']
                    distance = row['around_time']
                    message_2 += f"+ {around} - thời gian di chuyển: {distance}\n"
                dispatcher.utter_message(text=message_2)
        else:
            dispatcher.utter_message(text="Xin lỗi, hiện tại khách sạn này không có trong hồ sơ của tôi")
        return []

class ActionDijkstra(Action):
    def name(self) -> Text:
        return "action_dijkstra"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        path_hotel = '/Users/DELL/RASA/Hotel_Travel/Model_Tour/graph_hotel.json'
        path = '/Users/DELL/RASA/Hotel_Travel/Model_Tour/graph.json'

        with open(path_hotel, 'r') as file_hotel:
            graph_hotel = json.load(file_hotel)
        with open(path, 'r') as file:
            graph = json.load(file)

        corpus = list(graph.keys()) + list(graph_hotel.keys())
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        key = tracker.get_slot("name")
        num_waypoints = tracker.get_slot("num_waypoints")

        if key is not None and num_waypoints is not None and int(num_waypoints) < 50:
            key_vec = vectorizer.transform([key])
            similarity_scores = cosine_similarity(X, key_vec)

            if np.max(similarity_scores) < 0.7:
                dispatcher.utter_message(text="Không tìm thấy địa điểm phù hợp.")
            else:
                max_similarity_index = np.argmax(similarity_scores)
                start = corpus[max_similarity_index]
                nearest_attraction = graph.get(start, graph_hotel.get(start))
                if nearest_attraction:
                    path = self.tour_model(graph, start, int(num_waypoints) - 1)
                    if path:
                        table_data = self.prepare_trip_info(path)
                        formatted_path = ' --> '.join(path)
                        dispatcher.utter_message(text=f"Đây là lộ trình di chuyển tối ưu tôi tìm được cho bạn: {formatted_path}")
                        trip_details = self.generate_trip_details(table_data)
                        dispatcher.utter_message(text="Một số thông tin về lộ trình di chuyển:")
                        dispatcher.utter_message(text=trip_details)
                    else:
                        dispatcher.utter_message(text="Không thể tìm thấy đường đi phù hợp.")
                else:
                    dispatcher.utter_message(text="Không tìm thấy địa điểm gần nhất.")
        else:
            dispatcher.utter_message(text="Một số thông tin cần thiết bị thiếu hoặc số điểm đến vượt quá số lượng")
        return []


class ActionDijkstra(Action):
    def name(self) -> Text:
        return "action_dijkstra"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Đường dẫn tới file JSON
        path_hotel = '/Users/DELL/RASA/Hotel_Travel/Model_Tour/graph_hotel.json'
        path = '/Users/DELL/RASA/Hotel_Travel/Model_Tour/graph.json'

        # Đọc nội dung từ file JSON đầu tiên
        with open(path_hotel, 'r') as file_hotel:
            graph_hotel = json.load(file_hotel)
        
        # Đọc nội dung từ file JSON thứ hai
        with open(path, 'r') as file:
            graph = json.load(file)
        # TF-IDF
        corpus = list(graph.keys()) + list(graph_hotel.keys())
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        key = tracker.get_slot("name")
        num_waypoints = tracker.get_slot("num_waypoints")
        if key is not None and num_waypoints is not None and int(num_waypoints) < 50:
            key_vec = vectorizer.transform([key])
            similarity_scores = cosine_similarity(X,key_vec)
            if np.max(similarity_scores) < 0.7:
                dispatcher.utter_message(text="Không tìm thấy địa điểm này, vui lòng nhập lại")
            else: 
                max_similarity_index = np.argmax(similarity_scores)
                keys = list(graph.keys()) + list(graph_hotel.keys())
                start = keys[max_similarity_index]
                if start in graph_hotel:
                    nearest_attraction = graph_hotel.get(start)
                if start in graph:
                    nearest_attraction = start
                path = self.tour_model(graph, nearest_attraction, int(num_waypoints)-1)
                if path:
                    table_data = []
                    df = pd.read_csv(filepath_or_buffer='/Users/DELL/RASA/Hotel_Travel/Clean/Travel/clean_distance_travel.csv')
                    df_address = pd.read_csv(filepath_or_buffer='/Users/DELL/RASA/Hotel_Travel/Clean/Travel/clean_travel.csv')
                    for i in range(len(path)-1):
                        travel_point = path[i]
                        next_travel_point = path[i + 1]
                        matching_rows = df[(df['From'] == travel_point) & (df['To'] == next_travel_point)]
                    
                        if matching_rows.empty:
                            matching_rows = df[(df['From'] == next_travel_point) & (df['To'] == travel_point)]
                    
                        matching_time = matching_rows['Time'].values[0]
                        matching_distance = matching_rows['Distance'].values[0]
                        
                        table_data.append([travel_point, next_travel_point, matching_time, matching_distance])

                    for row in table_data:
                        distance = row[3]
                        if distance < 1:
                            row[3] = str(distance * 1000) + " m"
                        else:
                            row[3] = str(distance) + " km"
                    formatted_path = ' --> '.join(path)
                    dispatcher.utter_message(text=f"Đây là lộ trình di chuyển tối ưu tôi tìm được cho bạn: {formatted_path}")
                    dispatcher.utter_message(text="Một số thông tin về lộ trình di chuyển:")
                    matching_rows_address = df_address[df_address['name_travel'].isin(path)]
                    all_locations_info = ""
                    for index, row in matching_rows_address.iterrows():
                        matching_name = row['name_travel']
                        matching_address = row['address_travel']

                        all_locations_info += f'📍 {matching_name} - Vị trí: {matching_address}\n'
                
                    for data in table_data:
                        matching_travel_point = data[0]
                        matching_next_travel_point = data[1]
                        matching_time = data[2]
                        matching_distance = data[3]
                        all_locations_info += f"⏩ Từ vị trí {matching_travel_point} đến {matching_next_travel_point} mất khoảng {matching_time} và quãng đường là {matching_distance}\n"
                    dispatcher.utter_message(text=all_locations_info)    
        else:
            dispatcher.utter_message(text="Một số thông tin cần thiết bị thiếu hoặc số điểm đến vượt quá số lượng")
        return []
    
    def tour_model(self, graph: Dict[str, Any], start: str, num_waypoints: int) -> List[str]:
        # Khởi tạo khoảng cách: vô cùng cho tất cả các đỉnh trừ đỉnh bắt đầu
        distances = {node: (float('inf'), []) for node in graph}
        distances[start] = (0, [start])

        # Tạo hàng đợi ưu tiên và thêm đỉnh bắt đầu vào đó với số điểm đã thăm là 0 và danh sách điểm đã thăm là rỗng
        queue = [(0, start, 0, [start])]

        seen = {(start, 0)}  # Tạo một set lưu trữ các cặp (node, waypoints_visited)

        # Thực hiện thuật toán Dijkstra
        while queue:
            current_distance, current_node, waypoints_visited, visited_waypoints = heapq.heappop(queue)

            # Nếu đã thăm đủ số điểm tham quan theo yêu cầu, thì dừng lại
            if waypoints_visited == num_waypoints:  
                return visited_waypoints

            # Xét các đỉnh láng giềng
            for neighbor, weight in graph[current_node].items():
                # Kiểm tra xem điểm láng giềng này đã được thăm chưa
                if neighbor not in visited_waypoints:
                    distance = current_distance + weight

                    # Nếu tìm thấy đường ngắn hơn hoặc đỉnh láng giềng chưa được thăm
                    if (distance < distances[neighbor][0] or (neighbor, waypoints_visited + 1) not in seen):
                        distances[neighbor] = (distance, visited_waypoints + [neighbor])
                        heapq.heappush(queue, (distance, neighbor, waypoints_visited + 1, visited_waypoints + [neighbor]))
                        seen.add((neighbor, waypoints_visited + 1))  # Thêm vào set seen để không xử lý nút đó nữa

class ActionDefaultFallback(Action):

    def name(self) -> Text:
        return "action_default_fallback"

    async def run(
        self, 
        dispatcher: CollectingDispatcher, 
        tracker: Tracker, 
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(template="utter_default_fallback")

        return [UserUtteranceReverted()]