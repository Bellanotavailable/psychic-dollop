## Репозиторий для хранения и демонстрации лабораторных работ по ТРПО
- [`Lab4_polynomial`](https://github.com/Bellanotavailable/psychic-dollop/blob/main/l4/Lab4.py) 
    - создается класс полиномов `Polynomial`
    - в `__init__` передаётся список коэффициентов
    - методы должны возвращать привычную запись полинома, сумму двух полинов, разность, перемножение, деление уголком, возведение в степень, значение полинома в точке. Также он хотел добавить метод вычисления производной и определённого интеграла
- [`Lab5_Levenstein_distance_&_matrix_multiplication`](https://github.com/Bellanotavailable/psychic-dollop/blob/main/l5/Lab5.py) 
     - Eсть 2 строки у которых определены

        **в случае а**:
        операции вставки и удаления

        **в случае б**:
        перестановки символов между собой.

        <u>Найти</u> расстояние между строками, т. е минимальное количество операций (минимум из способов а и Б) чтобы превратить первую строку во вторую. 

        <u>Пример</u>: 'лось' и 'соль'. 

        В случае **а** - удаление букв л и с, вставка букв с и л. В случае **б** - поменять местами л и о , поменять местами л и с, поменять местами о и с. Количество операций в первом случае 2, во втором 3 , расстояние равно 2.


    - Дано множественное перемножение матриц, <u>нужно</u> расставить скобки в перемножении так, чтобы вычислительная сложность была наименьшей. 

        <u>На вход</u> - кортежи размеров матриц [m,n] О=О(m*n*k). 

        Также проверить возможность умножения их
