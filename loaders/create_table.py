import boto3

def create_dynamodb_table():
    dynamodb = boto3.resource('dynamodb', region_name='ap-southeast-1')

    table_name = 'ddb-htx-le-devizapp-imagehashes'

    existing_tables = dynamodb.meta.client.list_tables()['TableNames']
    if table_name in existing_tables:
        print(f"Table '{table_name}' already exists.")
        return dynamodb.Table(table_name)

    table = dynamodb.create_table(
        TableName=table_name,
        KeySchema=[
            {
                'AttributeName': 'url',
                'KeyType': 'HASH'  # Partition key
            },
        ],
        AttributeDefinitions=[
            {
                'AttributeName': 'url',
                'AttributeType': 'S'
            },
        ],
        ProvisionedThroughput={
            'ReadCapacityUnits': 5,
            'WriteCapacityUnits': 5
        }
    )

    # Wait until the table exists
    table.meta.client.get_waiter('table_exists').wait(TableName=table_name)

    print(f"Table '{table_name}' created successfully.")
    return table

if __name__ == "__main__":
    create_dynamodb_table()
