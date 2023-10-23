from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
from flask import Flask, render_template, request, jsonify
import pymysql
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')


app = Flask(__name__)


@app.route('/import-dataset', methods=['POST'])
def upload_csv():
    uploadedFile = request.files['formFile']
    if uploadedFile.filename != '':
        if uploadedFile.filename.endswith('.csv'):
            csv_path = os.path.join('uploads', uploadedFile.filename)
            uploadedFile.save(csv_path)

            df = pd.read_csv(csv_path, delimiter=';', header=0)

            mydb = pymysql.connect(
                host="localhost",
                user="root",
                password="",
                database="db_berdikari"
            )

            with mydb.cursor() as cursor:
                for index, row in df.iterrows():
                    sql = "INSERT IGNORE INTO penjualan (tgl, total_penjualan, produk_dilihat, total_pengunjung, pesanan_berhasil, pesanan_dibatalkan, pesanan_dikembalikan, pembeli) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
                    cursor.execute(sql, (row['tgl'], row['total_penjualan'], row['produk_dilihat'], row['total_pengunjung'],
                                   row['pesanan_berhasil'], row['pesanan_dibatalkan'], row['pesanan_dikembalikan'], row['pembeli']))
                mydb.commit()
                mydb.close()

            return jsonify({"message": "File CSV berhasil diunggah dan diproses"})
        else:
            return jsonify({"message": "Silahkan upload file yang memiliki format CSV"})
    else:
        return jsonify({"message": "Pilih file CSV terlebih dahulu"})


def format_rupiah(angka):
    formatted_angka = "Rp {:,.0f}".format(angka)

    # Replace the default thousands separator (,) with a dot (.)
    formatted_angka = formatted_angka.replace(",", ".")

    return formatted_angka


def format_angka_ribuan(angka):
    # Menggunakan f-string untuk mengonversi angka ke format dengan pemisah ribuan
    formatted_angka = f"{angka:,}"
    formatted_angka = formatted_angka.replace(",", ".")
    return formatted_angka


@app.route('/dataset')
def show_sales_table():
    mydb = pymysql.connect(
        host="localhost",
        user="root",
        password="",
        database="db_berdikari"
    )

    with mydb.cursor() as cursor:
        cursor.execute("SELECT * FROM penjualan")
        data = cursor.fetchall()

    sql_query = "SELECT tgl,total_penjualan,produk_dilihat,total_pengunjung,pesanan_berhasil,pesanan_dibatalkan,pesanan_dikembalikan,pembeli FROM `penjualan`"
    df = pd.read_sql(sql_query, con=mydb)
    df.drop('tgl', axis=1, inplace=True)

    plt.figure(figsize=(10, 8))

    sns.pairplot(data=df, x_vars=['produk_dilihat'], y_vars=[
                 'total_penjualan'], size=5, aspect=0.75)
    plt.savefig('static/assets/img/produk_dilihat.png')

    sns.pairplot(data=df, x_vars=['total_pengunjung'], y_vars=[
                 'total_penjualan'], size=5, aspect=0.75)
    plt.savefig('static/assets/img/total_pengunjung.png')

    sns.pairplot(data=df, x_vars=['pesanan_berhasil'], y_vars=[
                 'total_penjualan'], size=5, aspect=0.75)
    plt.savefig('static/assets/img/pesanan_Berhasil.png')

    sns.pairplot(data=df, x_vars=['pesanan_dibatalkan'], y_vars=[
                 'total_penjualan'], size=5, aspect=0.75)
    plt.savefig('static/assets/img/pesanan_dibatalkan.png')

    sns.pairplot(data=df, x_vars=['pesanan_dikembalikan'], y_vars=[
                 'total_penjualan'], size=5, aspect=0.75)
    plt.savefig('static/assets/img/pesanan_dikembalikan.png')

    sns.pairplot(data=df, x_vars=['pembeli'], y_vars=[
                 'total_penjualan'], size=5, aspect=0.75)
    plt.savefig('static/assets/img/pembeli.png')

    corr_matrix = df.corr()
    corr_matrix = corr_matrix.round(2)

    x = df.drop(columns='total_penjualan')
    y = df['total_penjualan']
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=4)
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)

    coef_dect = {
        'features': x.columns,
        'coef_values': lin_reg.coef_
    }

    akurasi = lin_reg.score(x_test, y_test) * 100
    persentase_akurasi = round(akurasi, 2)

    mydb.close()

    return render_template('dataset.html', data=data, format_rupiah=format_rupiah, format_angka_ribuan=format_angka_ribuan, akurasi=persentase_akurasi, corr_matrix=corr_matrix)


@app.route('/prediksi', methods=['POST'])
def predict_data():
    mydb = pymysql.connect(
        host="localhost",
        user="root",
        password="",
        database="db_berdikari"
    )

    with mydb.cursor() as cursor:
        cursor.execute("SELECT * FROM penjualan")
        data = cursor.fetchall()

    sql_query = "SELECT tgl,total_penjualan,produk_dilihat,total_pengunjung,pesanan_berhasil,pesanan_dibatalkan,pesanan_dikembalikan,pembeli FROM `penjualan`"
    df = pd.read_sql(sql_query, con=mydb)
    df.drop('tgl', axis=1, inplace=True)

    x = df.drop(columns='total_penjualan')
    y = df['total_penjualan']
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=4)
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)

    akurasi = lin_reg.score(x_test, y_test) * 100
    persentase_akurasi = round(akurasi, 2)

    variable1 = float(request.form['produk_dilihat'])
    variable2 = float(request.form['total_pengunjung'])
    variable3 = float(request.form['pesanan_berhasil'])
    variable4 = float(request.form['pesanan_dibatalkan'])
    variable5 = float(request.form['pesanan_dikembalikan'])
    variable6 = float(request.form['pembeli'])

    hasil = lin_reg.predict(
        [[variable1, variable2, variable3, variable4, variable5, variable6]])

    format_hasil = format_rupiah(hasil[0])

    hasil_json = {
        "produk_dilihat": int(variable1),
        "total_pengunjung": int(variable2),
        "pesanan_berhasil": int(variable3),
        "pesanan_dibatalkan": int(variable4),
        "pesanan_dikembalikan": int(variable5),
        "pembeli": int(variable6),
        "hasil": format_hasil,
    }

    mydb.close()

    return jsonify(hasil_json)


@app.route('/')
def data_total_pesanan():
    mydb = pymysql.connect(
        host="localhost",
        user="root",
        password="",
        database="db_berdikari"
    )

    with mydb.cursor() as cursorPesanan:
        cursorPesanan.execute(
            "SELECT SUM(pesanan_berhasil) AS total_pesanan FROM penjualan")
        result = cursorPesanan.fetchone()

    with mydb.cursor() as cursorPenjualan:
        cursorPenjualan.execute(
            "SELECT SUM(total_penjualan) AS total_penjualan FROM penjualan")
        resultPenjualan = cursorPenjualan.fetchone()

    with mydb.cursor() as cursorUser:
        cursorUser.execute(
            "SELECT SUM(pembeli) AS pembeli FROM penjualan")
        resultUser = cursorUser.fetchone()

    mydb.close()

    # Mengakses nilai pertama dalam tuple atau menggunakan nilai default 0 jika tidak ada hasil
    total_pesanan = result[0] if result else 0
    formatted_total_pesanan = format_angka_ribuan(total_pesanan)

    total_penjualan = resultPenjualan[0] if resultPenjualan else 0
    formatted_total_penjualan = format_rupiah(total_penjualan)

    total_pembeli = resultUser[0] if resultUser else 0
    formatted_total_pembeli = format_angka_ribuan(total_pembeli)

    return render_template('index.html', total_pesanan=formatted_total_pesanan, total_penjualan=formatted_total_penjualan, total_pembeli=formatted_total_pembeli)


@app.route('/data_chart')
def apexChart():
    data_pesanan = []
    data_penjualan = []
    data_pelanggan = []

    mydb = pymysql.connect(
        host="localhost",
        user="root",
        password="",
        database="db_berdikari"
    )

    try:
        with mydb.cursor() as cursor:
            for i in range(1, 13):
                cursor.execute(
                    "SELECT SUM(pesanan_berhasil) AS total_pesanan, "
                    "SUM(total_penjualan) AS total_penjualan, "
                    "SUM(pembeli) AS pembeli "
                    "FROM penjualan "
                    "WHERE MONTH(tgl) = %s AND YEAR(tgl) = %s",
                    (i, 2022)
                )
                result = cursor.fetchone()
                # Menggunakan indeks numerik 0
                data_pesanan.append(result[0] if result else 0)
                # Menggunakan indeks numerik 1
                data_penjualan.append(result[1] if result else 0)
                # Menggunakan indeks numerik 2
                data_pelanggan.append(result[2] if result else 0)

    except Exception as e:
        # Tangani kesalahan jika terjadi
        return jsonify({"error": str(e)}), 500
    finally:
        mydb.close()

    data_chart = {
        "data_pesanan": data_pesanan,
        "data_penjualan": data_penjualan,
        "data_pelanggan": data_pelanggan,
    }

    return jsonify(data_chart)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dataset')
def dataset():
    return render_template('dataset.html')


@app.route('/import-dataset')
def importDataset():
    return render_template('import-dataset.html')


@app.route('/prediksi')
def prediksi():
    return render_template('prediksi.html')


if __name__ == '__main__':
    app.run(debug=True, port=4000)
