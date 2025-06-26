import zipfile
import polars as pl
import pandas as pd
import io
import os
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
            'product_name': pl.Categorical,
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
DATATYPES_IMPROVED['day_duration'] = pl.UInt16


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
            'day_duration': pd.UInt16Dtype()
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
                                        filtered_df = filter_ancillary(df)
                                        df_list.append(filtered_df)
    return df_list


def filter_energy(df):
    # Apply filtering to the DataFrame
    filtered_df = df.filter(
        (pl.col('point_of_delivery_specific_location').is_in(['Mid-Columbia (Mid-C)', 'COB'])) &  # Only Mid-C and COB Trade Hub
        (pl.col('point_of_delivery_balancing_authority') == 'HUB') &  # Only Delivery to Trade Hub
        (pl.col('product_name') == 'ENERGY')  # Only Energy Product
    )
    return filtered_df


def filter_ancillary(df):
    # Apply filering to the DataFrame
    filtered_df = df.filter(
        (pl.col('product_name').is_in(['CAPACITY', 'REGULATION & FREQUENCY RESPONSE', 'PRIMARY FREQUENCY RESPONSE']))
    )
    return filtered_df
   
def fix_data_format(df, chunk_size=10000):
    # Fix the Datatypes
    df = df.with_columns([
        pl.col('transaction_begin_date').str.to_datetime(format="%Y%m%d%H%M"),
        pl.col('transaction_end_date').str.to_datetime(format="%Y%m%d%H%M"),
        pl.col('standardized_quantity').cast(pl.Float32),
        pl.col('total_transaction_charge').cast(pl.Float32)
    ])
   
    # Fix standardized_quantity and standardized_price errors
    def fix_errors(row):
        if row['standardized_quantity'] is None:
            row['standardized_quantity'] = 0
        if row['standardized_quantity'] == 0 and row['transaction_quantity'] != 0 and row['rate_units'] == '$/MWH':
            row['standardized_quantity'] = row['transaction_quantity']
        if row['standardized_price'] is None and row['rate_units'] == '$/MWH':
            row['standardized_price'] = row['price']
        return row
   
    for start in range(0, df.height, chunk_size):
        end = min(start + chunk_size, df.height)
        chunk = df.slice(start, end - start)
        rows = []
        for row in chunk.iter_rows(named=True):
            fixed_row = fix_errors(row)
            rows.append(fixed_row)


        # Replace the original chunk with the fixed rows
        df = df.slice(0, start).vstack(pl.DataFrame(rows, schema=df.schema)).vstack(df.slice(end, df.height - end))


    return df


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
                    chunk_df = fix_data_format(chunk_df)
                    intermediate_file = os.path.join(temp_dir, f'intermediate_chunk_{i}.csv')
                    chunk_df.write_csv(intermediate_file)
                    intermediate_files.append(intermediate_file)
                   
                print('Data Format fixed.')


                final_df = concat_inter_files(intermediate_files)
                print('Final DataFrame concatenation complete.')


                # Write final DataFrame to CSV
                output_dir = r'c:\Users\LauC2\ancillary_no_breakdown'
                output_file_name = 'intermediate_ancillary_transactions_no_breakdown_' + outer_file[4:11] + '.csv'
                final_output_path = os.path.join(output_dir, output_file_name)
                final_df.write_csv(final_output_path)
                print('Conversion from DataFrame to CSV complete.')
               
            finally:
                remove_temp(intermediate_files, temp_dir)
           
def final_concat_energy():
    with pl.StringCache():
        directory = r'c:\Users\LauC2\energy_no_breakdown'
        output_directory = r'O:\POOL\PRIVATE\RISKMGMT\EQR Reporting\EQR Study'
        output_file_path = os.path.join(output_directory, 'final_energy_transactions_no_breakdown.csv')
       
        # Remove the output file if it already exists to start fresh
        if os.path.exists(output_file_path):
            os.remove(output_file_path)
        write_header = True
        print('Read intermediate no Breakdown Energy files.')
       
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
        directory = r'c:\Users\LauC2\ancillary_no_breakdown'
        output_directory = r'O:\POOL\PRIVATE\RISKMGMT\EQR Reporting\EQR Study'
        output_file_path = os.path.join(output_directory, 'final_ancillary_transactions_no_breakdown.csv')
       
        # Remove the output file if it already exists to start fresh
        if os.path.exists(output_file_path):
            os.remove(output_file_path)
        write_header = True
        print('Read intermediate no Breakdown Ancillary files.')
       
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
    output_directory = r'O:\POOL\PRIVATE\RISKMGMT\EQR Reporting\EQR Study'
    output_file_path = os.path.join(output_directory, 'final_ancillary_transactions_no_breakdown.csv')
    df = pl.read_csv(output_file_path, has_header = True, n_rows=10)
    display(df)
   
if __name__ == '__main__':
    # Remove Comment once all Quarter Files are ready.
    final_concat_energy()
    final_concat_ancillary()

