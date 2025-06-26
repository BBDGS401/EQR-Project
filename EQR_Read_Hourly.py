import holidays
import zipfile
import polars as pl
import pandas as pd
import io
import datetime
import os
import math
from IPython.display import display


DATATYPES = {
            'transaction_unique_id': pl.Utf8,
            'seller_company_name': pl.Utf8,
            'customer_company_name': pl.Utf8,
            'ferc_tariff_reference': pl.Utf8,
            'contract_service_agreement': pl.Utf8,
            'transaction_unique_identifier': pl.Utf8,
            'transaction_begin_date': pl.Utf8,
            'transaction_end_date': pl.Utf8,
            'trade_date': pl.UInt32,
            'exchange_brokerage_service': pl.Utf8,
            'type_of_rate': pl.Categorical,
            'time_zone':pl.Categorical,
            'point_of_delivery_balancing_authority': pl.Utf8,
            'point_of_delivery_specific_location': pl.Utf8,
            'class_name': pl.Categorical,
            'term_name': pl.Categorical,
            'increment_name': pl.Categorical,
            'increment_peaking_name': pl.Categorical,
            'product_name': pl.Utf8,
            'transaction_quantity': pl.Float32,
            'price': pl.Float32,
            'rate_units' : pl.Categorical,
            'standardized_quantity': pl.Utf8,
            'standardized_price': pl.Utf8,
            'total_transmission_charge': pl.Float32,
            'total_transaction_charge': pl.Utf8
        }
       
DATATYPES_IMPROVED = DATATYPES.copy()
# Update specific fields with their improved types
DATATYPES_IMPROVED['standardized_quantity'] = pl.Float32
DATATYPES_IMPROVED['total_transaction_charge'] = pl.Float32
DATATYPES_IMPROVED['hour_duration'] = pl.UInt16


DATATYPES_PANDAS = {
            'transaction_unique_id': pd.StringDtype(),
            'seller_company_name': pd.StringDtype(),
            'customer_company_name': pd.StringDtype(),
            'ferc_tariff_reference': pd.StringDtype(),
            'contract_service_agreement': pd.StringDtype(),
            'transaction_unique_identifier': pd.StringDtype(),
            'transaction_begin_date': pd.StringDtype(),
            'transaction_end_date': pd.StringDtype(),
            'trade_date': pd.Int32Dtype(),
            'exchange_brokerage_service': pd.StringDtype(),
            'type_of_rate': pd.CategoricalDtype(categories=['', 'Fixed', 'Formula', 'Electric Index', 'RTO/ISO']),
            'time_zone':pd.CategoricalDtype(categories=['AD','AP','AS','CD','CP','CS','ED','EP','ES','MD','MP','MS','PD','PP','PS']),
            'point_of_delivery_balancing_authority': pd.StringDtype(),
            'point_of_delivery_specific_location': pd.StringDtype(),
            'class_name': pd.CategoricalDtype(categories=['', 'F', 'NF', 'UP', 'BA', 'N/A']),
            'term_name': pd.CategoricalDtype(categories=['LT', 'ST', 'N/A']),
            'increment_name': pd.CategoricalDtype(categories=['', '5', '15', 'H', 'D', 'W', 'M', 'Y', 'N/A']),
            'increment_peaking_name': pd.CategoricalDtype(categories=['', 'FP', 'OP', 'P', 'N/A']),
            'product_name': pd.CategoricalDtype(categories=['BLACK START SERVICE', 'BOOKED OUT POWER', 'CAPACITY', 'CUSTOMER CHARGE',
                                                            'DIRECT ASSIGNMENT FACILITIES CHARGE', 'EMERGENCY ENERGY', 'ENERGY',
                                                            'ENERGY IMBALANCE', 'EXCHANGE', 'FUEL CHARGE', 'GENERATOR IMBALANCE',
                                                            'GRANDFATHERED BUNDLED', 'INTERCONNECTION AGREEMENT', 'MEMBERSHIP AGREEMENT',
                                                            'MUST RUN AGREEMENT', 'NEGOTIATED-RATE TRANSMISSION', 'NETWORK',
                                                            'NETWORK OPERATING AGREEMENT', 'OTHER', 'POINT-TO-POINT AGREEMENT',
                                                            'PRIMARY FREQUENCY RESPONSE', 'REACTIVE SUPPLY & VOLTAGE CONTROL',
                                                            'REAL POWER TRANSMISSION LOSS', 'REASSIGNMENT AGREEMENT',
                                                            'REGULATION & FREQUENCY RESPONSE', 'REQUIREMENTS SERVICE',
                                                            'SCHEDULE SYSTEM CONTROL & DISPATCH', 'SPINNING RESERVE', 'SUPPLEMENTAL RESERVE',
                                                            'SYSTEM OPERATING AGREEMENTS', 'TOLLING ENERGY', 'TRANSMISSION OWNERS AGREEMENT'
                                                            , 'UPLIFT']),
            'transaction_quantity': pd.Float32Dtype(),
            'price': pd.Float32Dtype(),
            'rate_units' : pd.CategoricalDtype(categories=['$/KV', '$/KVA', '$/KVR', '$/KW', '$/KWH', '$/KW-DAY', '$/KW-MO', '$/KW-WK', '$/KW-YR',
                                                           '$/MW', '$/MWH', '$/MW-DAY', '$/MW-MO', '$/MW-WK', '$/MW-YR', '$/MVAR-YR', '$/RKVA',
                                                           'CENTS', 'CENTS/KVR', 'CENTS/KWH', 'FLAT RATE']),
            'standardized_quantity': pd.Float32Dtype(),
            'standardized_price': pd.StringDtype(),
            'total_transmission_charge': pd.Float32Dtype(),
            'total_transaction_charge': pd.Float32Dtype(),
            'hour_duration': pd.UInt16Dtype()
        }


# Function to process each outer ZIP File
def process_zip_file(outer_file):
    df_list = []  
    if zipfile.is_zipfile(outer_file):  # Only considers Zipfiles
        with zipfile.ZipFile(outer_file) as zf_outer:
            # Read the first Layer of the Zipfile (by Quarter)
            for inner_file in zf_outer.namelist():
                with zf_outer.open(inner_file) as inner_file:
                    if zipfile.is_zipfile(inner_file):
                        with zipfile.ZipFile(inner_file) as zf_inner:
                            # Read the second Layer of the Zipfile (by Company)
                            for csvfile in zf_inner.namelist():
                                if csvfile.lower().endswith('transactions.csv'):
                                    with zf_inner.open(csvfile) as source:
                                        csv_data = source.read().decode('ISO-8859-1')
                                        df = pl.read_csv(io.StringIO(csv_data), has_header=True, schema = DATATYPES, null_values='', encoding='utf8')
                                        filtered_df = filter_energy(df)
                                        df_list.append(filtered_df)
    return df_list


def filter_energy(df):
    # Apply filtering to the DataFrame
    filtered_df = df.filter(
        (pl.col('point_of_delivery_specific_location').str.to_uppercase().is_in(['MID-COLUMBIA (MID-C)', 'COB'])) &  # Only Mid-C and COB Trade Hub
        (pl.col('point_of_delivery_balancing_authority').str.to_uppercase() == 'HUB') &  # Only Delivery to Trade Hub
        (pl.col('product_name').str.to_uppercase() == 'ENERGY')  # Only Energy Product
    )
    return filtered_df


def filter_ancillary(df):
    # Apply filering to the DataFrame
    filtered_df = df.filter(
        (pl.col('product_name').str.to_uppercase().is_in(['CAPACITY', 'REGULATION & FREQUENCY RESPONSE', 'PRIMARY FREQUENCY RESPONSE']))
    )
    return filtered_df


def NERC_holiday(year):
    us_holidays = holidays.UnitedStates(years=year)
    NERC_holidays = []
    for k, v in us_holidays.items():
        if v in {"New Year's Day", 'Memorial Day', 'Independence Day', 'Labor Day', 'Thanksgiving', 'Christmas Day'}:
            NERC_holidays.append(k)
    return NERC_holidays


# Breakdown a Transaction into a Hourly Format
def breakdown_to_hourly(df, chunk_size=10000):
    df = df.with_columns([
        pl.col('transaction_begin_date').str.to_datetime(format="%Y%m%d%H%M"),
        pl.col('transaction_end_date').str.to_datetime(format="%Y%m%d%H%M"),
        pl.col('standardized_quantity').cast(pl.Float32),
        pl.col('total_transaction_charge').cast(pl.Float32)
    ])
   
    expanded_rows = []
    for start in range(0, df.height, chunk_size):
        end = min(start + chunk_size, df.height)
        chunk = df.slice(start, end - start)
        for row in chunk.iter_rows(named=True):
            begin_date = row['transaction_begin_date']
            end_date = row['transaction_end_date']
            if row['standardized_quantity'] is None:
                row['standardized_quantity'] = 0
            if row['standardized_quantity'] == 0 and row['transaction_quantity'] != 0 and row['rate_units'] == '$/MWH':
                row['standardized_quantity'] = row['transaction_quantity']
            if row['standardized_price'] is None and row['rate_units'] == '$/MWH':
                row['standardized_price'] = row['price']        
            if begin_date != end_date:
                total_duration = (end_date - begin_date).total_seconds()
                current_date = begin_date
                rows_of_one_transaction = []
                while current_date < end_date:
                    if current_date == begin_date:
                        start_time = begin_date
                    else:
                        start_time = datetime.datetime.combine(current_date.date(), datetime.time(current_date.hour, 0))


                    if current_date.date() == end_date.date() and current_date.hour == end_date.hour:
                        end_time = end_date
                    else:
                        end_time = datetime.datetime.combine(current_date.date(), datetime.time(current_date.hour, 59))


                    if peaking_hour_error(row['increment_peaking_name'], start_time):
                        total_duration -= 3600
                        current_date += datetime.timedelta(hours=1)
                        continue
                    current_date_duration = (end_time - start_time).total_seconds()


                    new_row = row.copy()
                    new_row['transaction_begin_date'] = start_time.strftime('%Y/%m/%d %H:%M')
                    new_row['transaction_end_date'] = end_time.strftime('%Y/%m/%d %H:%M')
                    new_row['transaction_quantity'] = row['transaction_quantity'] * current_date_duration / total_duration
                    new_row['standardized_quantity'] = row['standardized_quantity'] * current_date_duration / total_duration
                    new_row['total_transaction_charge'] = new_row['transaction_quantity'] * new_row['price'] + new_row['total_transmission_charge']
                   
                    rows_of_one_transaction.append(new_row)
                    current_date += datetime.timedelta(hours=1)
               
                for one_row in rows_of_one_transaction:
                    one_row['hour_duration'] = math.ceil(total_duration / 3600) # Convert Duration to Hours
                    expanded_rows.append(one_row)
            else:
                new_row = row.copy()
                new_row['transaction_begin_date'] = begin_date.strftime('%Y/%m/%d %H:%M')
                new_row['transaction_end_date'] = end_date.strftime('%Y/%m/%d %H:%M')
                new_row['hour_duration'] = 1 # Same exact Datetime, so must be within a Hour
                expanded_rows.append(new_row)
    return pl.DataFrame(expanded_rows, schema = DATATYPES_IMPROVED)


# Determine if there is any Discrepencies between increment_peaking_name and Transaction Time
def peaking_hour_error(increment_peaking_name, start_time):
    # When breakdown to hourly, we should exclude peak Hours if Transaction is off-peak
    if (increment_peaking_name == 'OP'):
        return (start_time.weekday() != 6 and
        start_time.hour >= 6 and start_time.hour <= 21 and
        start_time.date() not in NERC_holiday(start_time.year))
    # When breakdown to hourly, we should exclude off-peak Hours if Transaction is peak    
    elif (increment_peaking_name == 'P'):
        return (start_time.hour < 6 or start_time.hour > 21 or
        start_time.weekday() == 6 or
        start_time.date() in NERC_holiday(start_time.year))
    else:
        return False


# Concatenate intermediate Files
def concat_inter_files(intermediate_files):
    final_dfs = []
    for file in intermediate_files:
        final_dfs.append(pl.read_csv(file, schema = DATATYPES_IMPROVED))
    return pl.concat(final_dfs)


# Create temporary Directory
def create_temp_dir(temp_dir):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    print('Temporary Directory created.')


# Clean up intermediate, temporary Files and Directory
def remove_temp(intermediate_files, temp_dir):
    for file in intermediate_files:
        os.remove(file)
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)
    print('All temp Files and temp Directories removed.')


def main(year_quarter):
        # Read all Zipfile from 2014 Q1 to 2024 Q1
    directory = r'O:\POOL\PRIVATE\RISKMGMT\EQR Reporting\EQR Study\Data_Files'
    os.chdir(directory)  # Change Directory


    with pl.StringCache():
        # Data Types for CSV Files
        outer_file = 'CSV_' + year_quarter + '.zip'
        print('Start processing ', outer_file, '.', sep='')
        df_list = process_zip_file(outer_file)
        print("All Data in", outer_file[4:11], 'are filtered.')
        if len(df_list) > 0:
            df_quarter = pl.concat(df_list)  # Concatenates Dataframes in a Quarter
            print('First DataFrame Concatenation complete.')


            # Create a temporary directory to store intermediate results
            temp_dir = r'c:\Users\LauC2\Temp'
            create_temp_dir(temp_dir)


            chunk_size = 10000
            intermediate_files = []


            try:
                for i, chunk in enumerate(df_quarter.iter_slices(chunk_size)):
                    chunk_df = pl.DataFrame(chunk, schema = DATATYPES)
                    hourly_chunk_df = breakdown_to_hourly(chunk_df)
                    intermediate_file = os.path.join(temp_dir, f'intermediate_chunk_{i}.csv')
                    hourly_chunk_df.write_csv(intermediate_file)
                    intermediate_files.append(intermediate_file)
                   
                print('Hourly Breakdown complete.')


                final_df = concat_inter_files(intermediate_files)
                print('Final DataFrame concatenation complete.')


                # Write final DataFrame to CSV
                output_dir = r'c:\Users\LauC2\energy_hourly'
                output_file_name = 'intermediate_energy_transactions_hourly_' + outer_file[4:11] + '.csv'
                final_output_path = os.path.join(output_dir, output_file_name)
                final_df.write_csv(final_output_path)
                print('Conversion from DataFrame to CSV complete.')
               
            finally:
                remove_temp(intermediate_files, temp_dir)
           
def final_concat_energy():
    with pl.StringCache():
        directory = r'c:\Users\LauC2\energy_hourly'
        output_file_path = os.path.join(directory, 'final_energy_transactions_hourly_2.csv')
       
        # Remove the output file if it already exists to start fresh
        if os.path.exists(output_file_path):
            os.remove(output_file_path)
        write_header = True
        print('Read intermediate hourly Energy files.')
       
        with open(output_file_path, mode="ab") as output_file:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                df = pl.read_csv(file_path, has_header=True, schema = DATATYPES_IMPROVED, null_values='', encoding='utf8')
                # Write the dataframe to the output CSV
                df.write_csv(output_file, include_header=write_header)
                write_header = False
                print(f'Appended {file}.')  
        print('Contatenation complete.')
       
def final_concat_ancillary():
    with pl.StringCache():
        directory = r'c:\Users\LauC2\ancillary_hourly'
        output_directory = r'O:\POOL\PRIVATE\RISKMGMT\EQR Reporting\EQR Study'
        output_file_path = os.path.join(output_directory, 'final_ancillary_transactions_hourly_2.csv')
       
        # Remove the output file if it already exists to start fresh
        if os.path.exists(output_file_path):
            os.remove(output_file_path)
        write_header = True
        print('Read intermediate hourly Ancillary files.')
       
        with open(output_file_path, mode="ab") as output_file:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                df = pl.read_csv(file_path, has_header=True, schema=DATATYPES_IMPROVED, null_values='', encoding='utf8')
                # Write the dataframe to the output CSV
                df.write_csv(output_file, include_header=write_header)
                write_header = False
                print(f'Appended {file}.')            
        print('Appending complete.')
       
def test():
    directory = r'c:\Users\LauC2\energy_hourly'
    os.chdir(directory)
    df = pl.read_csv('final_energy_transactions_hourly.csv', n_rows=10, has_header=True)
    display(df)        


if __name__ == '__main__':
    # Remove Comment once all Quarter Files are ready.
    final_concat_energy()

