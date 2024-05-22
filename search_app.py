from kivy.app import App
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.metrics import dp, sp
from kivy.uix.image import Image
import time
from nltk.stem import PorterStemmer
from Posting import Posting
import pickle

# Setting window background to white
Window.clearcolor = (1, 1, 1, 1)

FOLDERNAME = "ANALYST"

class SearchScreen(Screen):
    def build(self):
        layout = AnchorLayout(anchor_x='center', anchor_y='center')

        box_layout = BoxLayout(orientation='vertical', size_hint=(0.8, 0.5), spacing=dp(10))

        # Logo
        logo = Image(source='UCI_Anteaters_logo.png', size_hint=(None, None), size=(dp(200), dp(100)))
        logo_layout = AnchorLayout(anchor_x='center', anchor_y='center')
        logo_layout.add_widget(logo)
        box_layout.add_widget(logo_layout)

        # Query input
        self.query_input = TextInput(hint_text='Enter your query', multiline=False, size_hint=(1, None), height=dp(40),
                                     padding=[dp(10), dp(10), dp(10), dp(10)], font_size=sp(20), background_color=[1, 1, 1, 1],
                                     foreground_color=[0, 0, 0, 1])
        box_layout.add_widget(self.query_input)

        # Results count input
        self.count_input = TextInput(hint_text='Number of results', multiline=False, input_filter='int', size_hint=(1, None),
                                     height=dp(40), padding=[dp(10), dp(10), dp(10), dp(10)], font_size=sp(20),
                                     background_color=[1, 1, 1, 1], foreground_color=[0, 0, 0, 1])
        box_layout.add_widget(self.count_input)

        # Search button
        search_button = Button(text='Search', size_hint=(1, None), height=dp(50), background_color=[0.26, 0.42, 0.96, 1],
                               font_size=sp(20), color=[1, 1, 1, 1])
        search_button.bind(on_press=self.on_search)
        box_layout.add_widget(search_button)

        layout.add_widget(box_layout)
        self.add_widget(layout)

    def on_search(self, instance):
        query = self.query_input.text
        count = self.count_input.text

        if not query or not count:
            return

        count = int(count)
        self.manager.current = 'results'
        self.manager.get_screen('results').search(query, count)

class ResultsScreen(Screen):
    def build(self):
        layout = BoxLayout(orientation='vertical', padding=dp(10), spacing=dp(10))

        # Back button
        back_button = Button(text='Back', size_hint=(1, None), height=dp(50), background_color=[0.26, 0.42, 0.96, 1],
                             font_size=sp(20), color=[1, 1, 1, 1])
        back_button.bind(on_press=self.go_back)
        layout.add_widget(back_button)

        # Results display
        self.results_layout = GridLayout(cols=1, size_hint_y=None, spacing=dp(20), padding=[dp(10), dp(10), dp(10), dp(10)])
        self.results_layout.bind(minimum_height=self.results_layout.setter('height'))

        scroll_view = ScrollView(size_hint=(1, 1))
        scroll_view.add_widget(self.results_layout)

        layout.add_widget(scroll_view)

        # Pagination buttons
        pagination_layout = BoxLayout(orientation='horizontal', size_hint=(1, None), height=dp(50), spacing=dp(10))

        self.prev_button = Button(text='Previous', size_hint=(None, None), size=(dp(100), dp(50)), background_color=[0.26, 0.42, 0.96, 1],
                                  font_size=sp(20), color=[1, 1, 1, 1])
        self.prev_button.bind(on_press=self.prev_page)
        pagination_layout.add_widget(self.prev_button)

        self.next_button = Button(text='Next', size_hint=(None, None), size=(dp(100), dp(50)), background_color=[0.26, 0.42, 0.96, 1],
                                  font_size=sp(20), color=[1, 1, 1, 1])
        self.next_button.bind(on_press=self.next_page)
        pagination_layout.add_widget(self.next_button)

        layout.add_widget(pagination_layout)

        self.add_widget(layout)

    def go_back(self, instance):
        self.manager.current = 'search'

    def search(self, query, count):
        self.query = query
        self.count = count
        self.page = 0

        ps = PorterStemmer()
        start = time.time()

        # Process query
        total_queries = [ps.stem(quer) for quer in query.split()]

        # Retrieve postings
        total_postings = []
        with open(f"{FOLDERNAME}_merged_inverted_index.pkl", "rb") as inverted_index_file:
            inverted_index = pickle.load(inverted_index_file)
            print(inverted_index)
            total_postings = [inverted_index[query] if query in inverted_index else [] for query in total_queries]
            print(total_postings)
        
        self.final_postings = self.merge_posting_lists(total_postings) if len(total_postings) > 1 else total_postings[0]
        self.final_postings.sort(key=lambda x: x.tf_idf, reverse=True)

        end = time.time()
        self.search_time = end - start
        self.display_results()

    def merge_posting_lists(self, total_postings):
        if not total_postings:
            return []
        total_postings.sort(key=len)
        final_posting = total_postings[0]
        for i in range(1, len(total_postings)):
            final_posting = self.intersect_postings(final_posting, total_postings[i])
            if not final_posting:
                break
        return final_posting

    def intersect_postings(self, posting1, posting2):
        merged = []
        i, j = 0, 0
        while i < len(posting1) and j < len(posting2):
            if posting1[i].docID == posting2[j].docID:
                merged.append(Posting(posting1[i].docID, posting1[i].freqCount + posting2[j].freqCount))
                i += 1
                j += 1
            elif posting1[i].docID < posting2[j].docID:
                i += 1
            else:
                j += 1
        return merged

    def display_results(self):
        self.results_layout.clear_widgets()

        if not self.final_postings:
            self.results_layout.add_widget(Label(text="No results found", font_size=sp(18), color=[1, 0, 0, 1]))
            return

        self.results_layout.add_widget(Label(text=f"Time to search: {self.search_time:.4f} seconds", font_size=sp(16), color=[0, 0, 0, 1]))
        self.results_layout.add_widget(Label(text="Top results:", font_size=sp(18), color=[0, 0, 0, 1]))

        start_index = self.page * self.count
        end_index = start_index + self.count

        with open(f"{FOLDERNAME}_url_map.pkl", "rb") as url_map_file:
            for i in range(start_index, min(end_index, len(self.final_postings))):
                doc_id = str(self.final_postings[i].docID)
                title, url, description = url_map_file.get(str(doc_id), ["Title Not Found", "URL Not Found", "Summary Not Found"])
                result_box = BoxLayout(orientation='vertical', padding=[dp(10), dp(10), dp(10), dp(10)], spacing=dp(5))
                result_box.add_widget(Label(text=f"#{i+1}: {title[0]}", font_size=sp(16), color=[0, 0, 1, 1], text_size=(self.width-dp(20), None), halign='left', valign='middle'))
                result_box.add_widget(Label(text=url, font_size=sp(14), color=[0, 0, 0, 1], text_size=(self.width-dp(20), None), halign='left', valign='middle'))
                result_box.add_widget(Label(text=description, font_size=sp(14), color=[0, 0, 0, 1], text_size=(self.width-dp(20), None), halign='left', valign='middle'))
                self.results_layout.add_widget(result_box)

        # Manage pagination buttons
        self.prev_button.disabled = self.page == 0
        self.next_button.disabled = end_index >= len(self.final_postings)

    def prev_page(self, instance):
        if self.page > 0:
            self.page -= 1
            self.display_results()

    def next_page(self, instance):
        if (self.page + 1) * self.count < len(self.final_postings):
            self.page += 1
            self.display_results()

class SearchApp(App):
    def build(self):
        sm = ScreenManager()
        search_screen = SearchScreen(name='search')
        search_screen.build()
        results_screen = ResultsScreen(name='results')
        results_screen.build()
        sm.add_widget(search_screen)
        sm.add_widget(results_screen)
        return sm

if __name__ == "__main__":
    SearchApp().run()
