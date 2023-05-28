# gauss
Bu kod örneğinde, GaussianMixtureModel sınıfı, GMM'nin temel özelliklerini ve yöntemlerini içeren bir sınıftır. Train metodu, verilen veri kümesini kullanarak GMM'yi eğitmek için EM (Expectation-Maximization) algoritmasını uygular. EM algoritması, iteratif olarak beklenti ve maksimizasyon adımlarını tekrarlayarak GMM'nin parametrelerini tahmin eder.

Predict metodu, verilen bir veri noktasının GMM tarafından tahmin edilen olasılığını hesaplar. Tahmin, her bileşenin ağırlığı ve ilgili Gauss dağılımının yoğunluğunun ağırlıklı toplamı olarak hesaplanır.

Kod örneğinde, veri kümesi data olarak tanımlanır ve GMM modeli gmm olarak oluşturulur. Ardından, gmm.Train(data, numIterations) çağrısıyla veri kümesi eğitilir ve ardından testData içindeki veri noktaları üzerinde tahminler yapılır.

Sonuçlar, her veri noktası için ekrana yazdırılır ve kullanıcıdan bir girdi almak için Console.ReadLine() kullanılır
