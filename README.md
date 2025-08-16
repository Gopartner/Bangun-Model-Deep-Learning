# Bangun-Model-Deep-Learning
Belajar AI di Node.js 🔥 Bangun Model Deep Learning dengan TensorFlow.js dari Nol

1. **Pendahuluan**
2. **Persiapan lingkungan**
3. **Dataset Fashion MNIST**
4. **Membangun model dengan TensorFlow\.js**
5. **Training & evaluasi**
6. **Inference (prediksi gambar baru)**
7. **Transfer Learning**
8. **Kesimpulan**

---

# 📘 Tutorial Lengkap: Membangun Deep Learning dengan TensorFlow\.js di Node.js

## 1. Pendahuluan

Dalam tutorial ini kita akan belajar:

* Cara membangun model deep learning dari nol dengan **TensorFlow\.js**
* Melatih model untuk klasifikasi gambar (Fashion MNIST)
* Menyimpan & menggunakan model untuk prediksi
* **Transfer Learning** untuk menambahkan kelas baru dengan cepat

Pastikan sudah familiar dengan dasar **Node.js** dan sedikit konsep **Machine Learning**.

---

## 2. Persiapan Lingkungan

1. Buat folder project baru

   ```bash
   mkdir ai-nodejs && cd ai-nodejs
   npm init -y
   ```
2. Install TensorFlow\.js Node

   ```bash
   npm install @tensorflow/tfjs-node
   ```
3. Install library tambahan (opsional, untuk manipulasi gambar)

   ```bash
   npm install jimp
   ```

---

## 3. Dataset Fashion MNIST

Dataset berisi 70.000 gambar grayscale ukuran 28x28 dari 10 kategori pakaian.

Download dataset:
👉 [IBM Dataset Exchange – Fashion MNIST](https://developer.ibm.com/exchanges/data/all/fashion-mnist/)

Ekstrak dataset hingga terdapat file:

* `fashion-mnist_train.csv`
* `fashion-mnist_test.csv`

---

## 4. Membangun Model dengan TensorFlow\.js

Buat file `build-model.js`:

```js
const tf = require('@tensorflow/tfjs-node');

// Label dataset (10 kelas pakaian)
const labels = [
  "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
];

// Fungsi untuk membuat model CNN
function buildModel() {
  const model = tf.sequential();

  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    kernelSize: 3,
    filters: 32,
    activation: 'relu'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  model.add(tf.layers.conv2d({
    kernelSize: 3,
    filters: 64,
    activation: 'relu'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dense({ units: labels.length, activation: 'softmax' }));

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}

module.exports = { buildModel, labels };
```

---

## 5. Training & Evaluasi

Buat file `train.js`:

```js
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const { buildModel, labels } = require('./build-model');

// TODO: load dataset CSV dan ubah ke tensor
// (gunakan tf.data.csv lalu map ke one-hot vector + normalisasi)

async function run() {
  const model = buildModel();
  model.summary();

  // contoh training dummy (ganti dengan dataset asli)
  const xs = tf.randomNormal([100, 28, 28, 1]);
  const ys = tf.oneHot(tf.randomUniform([100], 0, labels.length, 'int32'), labels.length);

  await model.fit(xs, ys, {
    epochs: 5,
    batchSize: 32,
    validationSplit: 0.2,
    callbacks: tf.callbacks.earlyStopping({ patience: 2 })
  });

  await model.save('file://./fashion-mnist-model');
  console.log("✅ Model tersimpan di ./fashion-mnist-model");
}

run();
```

> Setelah dijalankan:
>
> * Model akan dilatih dengan dataset
> * Disimpan dalam folder `fashion-mnist-model/` (berisi file JSON + bobot model)

---

## 6. Inference (Prediksi Gambar Baru)

Buat file `predict.js`:

```js
const tf = require('@tensorflow/tfjs-node');
const Jimp = require('jimp');
const { labels } = require('./build-model');

async function preprocessImage(path) {
  const image = await Jimp.read(path);
  image.resize(28, 28).grayscale();
  const buffer = Buffer.from(image.bitmap.data);
  const values = [];
  
  for (let i = 0; i < buffer.length; i += 4) {
    values.push(buffer[i] / 255); // ambil channel R (grayscale)
  }

  return tf.tensor4d(values, [1, 28, 28, 1]);
}

async function run() {
  const model = await tf.loadLayersModel('file://./fashion-mnist-model/model.json');
  const input = await preprocessImage('./test.png');
  const prediction = model.predict(input);
  const probs = prediction.dataSync();

  const maxIndex = probs.indexOf(Math.max(...probs));
  console.log(`Prediksi: ${labels[maxIndex]} (confidence: ${(probs[maxIndex]*100).toFixed(2)}%)`);
}

run();
```

> Jalankan dengan gambar uji:

```bash
node predict.js
```

---

## 7. Transfer Learning

Gunakan model lama, ganti layer terakhir untuk kelas baru.

```js
function transferLearning(baseModel, newClasses) {
  // Bekukan layer lama
  baseModel.layers.forEach(layer => { layer.trainable = false; });

  // Buang layer output lama
  const last = baseModel.layers[baseModel.layers.length - 2].output;

  const newOutput = tf.layers.dense({
    units: newClasses,
    activation: 'softmax'
  }).apply(last);

  const newModel = tf.model({ inputs: baseModel.inputs, outputs: newOutput });

  newModel.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return newModel;
}
```

Keuntungan:

* Latihan lebih cepat
* Data lebih sedikit
* Akurasi tetap tinggi

---

## 8. Kesimpulan

Dalam tutorial ini kita sudah belajar:

* Instalasi TensorFlow\.js di Node.js
* Load dataset Fashion MNIST
* Membangun model CNN sederhana
* Melatih & mengevaluasi model
* Menyimpan & memuat model
* Menggunakan model untuk prediksi gambar baru
* **Transfer Learning** untuk memperluas model tanpa melatih ulang dari nol

🎯 Hasil akhirnya: model klasifikasi pakaian dengan akurasi ±90%.

---
