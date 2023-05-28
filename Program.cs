using System;

namespace GMMExample
{
    class Program
    {
        static void Main(string[] args)
        {
            // Veri kümesi oluşturma
            double[] data = { 1.2, 1.4, 2.5, 2.7, 3.8, 4.0 };

            // GMM modelinin oluşturulması
            int numClusters = 2; // Karışım sayısı (bileşen sayısı)
            int numIterations = 100; // İterasyon sayısı

            // GMM modelini oluştur
            var gmm = new GaussianMixtureModel(numClusters);

            // Veri kümesini GMM modeline uygula
            gmm.Train(data, numIterations);

            // GMM ile tahmin yapma
            double[] testData = { 1.0, 2.8, 4.5 }; // Tahmin yapılacak veri noktaları
            foreach (var point in testData)
            {
                double probability = gmm.Predict(point);
                Console.WriteLine($"Veri Noktası: {point}, Olasılık: {probability}");
            }

            Console.ReadLine();
        }
    }

    // Gaussian Karışım Modeli sınıfı
    class GaussianMixtureModel
    {
        private int numClusters; // Karışım sayısı (bileşen sayısı)
        private double[] weights; // Karışım ağırlıkları
        private Normal[] distributions; // Gaussian dağılımları

        public GaussianMixtureModel(int numClusters)
        {
            this.numClusters = numClusters;
            weights = new double[numClusters];
            distributions = new Normal[numClusters];
        }

        public void Train(double[] data, int numIterations)
        {
            int dataLength = data.Length;

            // Başlangıç değerleriyle karışım ağırlıkları ve Gaussian dağılımları oluştur
            for (int i = 0; i < numClusters; i++)
            {
                weights[i] = 1.0 / numClusters; // Eşit ağırlıklarla başla
                double mean = data[i % dataLength]; // Veri noktalarını başlangıç ortalamaları olarak seç
                double variance = 1.0; // Başlangıç varyansı
                distributions[i] = new Normal(mean, Math.Sqrt(variance));
            }

            // GMM eğitim algoritmasını burada uygula (EM algoritması)
            for (int iter = 0; iter < numIterations; iter++)
            {
                // Expectation (Beklenti) adımı: Her veri noktasının hangi bileşene ait olduğunu tahmin et
                double[,] responsibilities = new double[dataLength, numClusters];
                for (int i = 0; i < dataLength; i++)
                {
                    double sum = 0.0;
                    for (int j = 0; j < numClusters; j++)
                    {
                        responsibilities[i, j] = weights[j] * distributions[j].Density(data[i]);
                        sum += responsibilities[i, j];
                    }
                    for (int j = 0; j < numClusters; j++)
                    {
                        responsibilities[i, j] /= sum;
                    }
                }

                // Maximization (Maksimizasyon) adımı: Parametreleri güncelle
                for (int j = 0; j < numClusters; j++)
                {
                    double sumResponsibilities = 0.0;
                    double sumData = 0.0;
                    double sumSquaredData = 0.0;

                    for (int i = 0; i < dataLength; i++)
                    {
                        sumResponsibilities += responsibilities[i, j];
                        sumData += responsibilities[i, j] * data[i];
                        sumSquaredData += responsibilities[i, j] * data[i] * data[i];
                    }

                    weights[j] = sumResponsibilities / dataLength;
                    double mean = sumData / sumResponsibilities;
                    double variance = sumSquaredData / sumResponsibilities - mean * mean;

                    distributions[j] = new Normal(mean, Math.Max(variance, double.Epsilon));
                }
            }
        }

        public double Predict(double dataPoint)
        {
            double probability = 0.0;

            for (int j = 0; j < numClusters; j++)
            {
                probability += weights[j] * distributions[j].Density(dataPoint);
            }

            return probability;
        }
    }

    class Normal
    {
        private double mean;
        private double variance;

        public Normal(double mean, double variance)
        {
            this.mean = mean;
            this.variance = variance;
        }

        public double Density(double x)
        {
            double exponent = -(Math.Pow(x - mean, 2) / (2 * variance));
            double coefficient = 1 / Math.Sqrt(2 * Math.PI * variance);
            return coefficient * Math.Exp(exponent);
        }
    }
}
