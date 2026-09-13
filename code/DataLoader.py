import pandas as pd
import os

class DataLoader:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.data = {}
        self._load_csvs()

    def _load_csvs(self):
        """Loads all CSV files from the dataset directory, excluding sample_requests.csv."""
        for filename in os.listdir(self.dataset_dir):
            if filename.endswith('.csv') and filename != 'sample_requests.csv':
                filepath = os.path.join(self.dataset_dir, filename)
                table_name = filename[:-4]
                self.data[table_name] = pd.read_csv(filepath)

    def get_exchange_rate(self, date: str, from_currency: str, to_currency: str) -> float:
        """Looks up the exact exchange rate for a given date and currency pair."""
        if from_currency == to_currency:
            return 1.0
            
        rates_df = self.data.get('exchange_rates')
        if rates_df is None:
            raise ValueError("exchange_rates.csv is not loaded.")
            
        # Find exact match
        match = rates_df[(rates_df['rate_date'] == date) & 
                         (rates_df['from_currency'] == from_currency) & 
                         (rates_df['to_currency'] == to_currency)]
                         
        if match.empty:
            raise ValueError(f"Exchange rate not found for date='{date}', from='{from_currency}', to='{to_currency}'.")
            
        return float(match.iloc[0]['rate'])

    def get_normalized_context(self, request_id: str) -> dict:
        """
        Gathers request context and normalizes all financial amounts to the user's home currency.
        Returns a well-organized dictionary.
        """
        # 1. Get the Request
        requests_df = self.data.get('requests')
        req_match = requests_df[requests_df['request_id'] == request_id]
        if req_match.empty:
            raise ValueError(f"Request ID '{request_id}' not found in requests.csv.")
        request_row = req_match.iloc[0].to_dict()
        user_id = request_row['user_id']
        
        # 2. Get User Profile & Home Currency
        profiles_df = self.data.get('financial_profiles')
        prof_match = profiles_df[profiles_df['user_id'] == user_id]
        if prof_match.empty:
            raise ValueError(f"User ID '{user_id}' not found in financial_profiles.csv.")
        user_profile = prof_match.iloc[0].to_dict()
        home_currency = user_profile['home_currency']
        
        # 3. Get Financial Events matching user_id
        events_df = self.data.get('financial_events')
        user_events = events_df[events_df['user_id'] == user_id].copy() if events_df is not None else pd.DataFrame()
        
        # 4. Get Payment Options matching request_id
        options_df = self.data.get('request_payment_options')
        payment_options = options_df[options_df['request_id'] == request_id].copy() if options_df is not None else pd.DataFrame()
        
        # 5. Get Messages and Images matching request_id, user_id, or related_event_id
        event_ids = user_events['event_id'].tolist() if not user_events.empty else []
        
        messages_df = self.data.get('messages')
        if messages_df is not None and not messages_df.empty:
            mask = (
                (messages_df.get('request_id') == request_id) |
                (messages_df.get('user_id') == user_id) |
                (messages_df.get('related_event_id').isin(event_ids))
            )
            messages = messages_df[mask].copy()
        else:
            messages = pd.DataFrame()
            
        images_df = self.data.get('images')
        if images_df is not None and not images_df.empty:
            mask = (
                (images_df.get('request_id') == request_id) |
                (images_df.get('user_id') == user_id) |
                (images_df.get('related_event_id').isin(event_ids))
            )
            images = images_df[mask].copy()
        else:
            images = pd.DataFrame()
        
        # 6. Normalize Currencies in Financial Events
        normalized_events = []
        for _, event in user_events.iterrows():
            event_dict = event.to_dict()
            
            if pd.notna(event_dict.get('amount')) and pd.notna(event_dict.get('currency')):
                evt_currency = event_dict['currency']
                
                if evt_currency != home_currency:
                    # Problem Statement: Use settlement_date for cash events.
                    # If settlement_date is unavailable, fallback to event_date.
                    date_for_rate = event_dict.get('settlement_date')
                    if pd.isna(date_for_rate):
                        date_for_rate = event_dict.get('event_date')
                        
                    if pd.isna(date_for_rate):
                        raise ValueError(f"No date found for event {event_dict.get('event_id')} to perform currency conversion.")
                        
                    rate = self.get_exchange_rate(date_for_rate, evt_currency, home_currency)
                    
                    event_dict['amount'] = event_dict['amount'] * rate
                    event_dict['currency'] = home_currency
                    
                    # Also normalize minimum_allowed_amount if present
                    if pd.notna(event_dict.get('minimum_allowed_amount')):
                        event_dict['minimum_allowed_amount'] = event_dict['minimum_allowed_amount'] * rate
                        
            normalized_events.append(event_dict)
            
        # Note: requests and request_payment_options are already in the user's home_currency per problem statement.
        
        # 7. Return Clean Dictionary
        return {
            'request': request_row,
            'user_profile': user_profile,
            'financial_events': normalized_events,
            'request_payment_options': payment_options.to_dict(orient='records'),
            'messages': messages.to_dict(orient='records'),
            'images': images.to_dict(orient='records')
        }
