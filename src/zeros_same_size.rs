use ndarray::*;

trait SqrtArr {
    fn sqrt(&self) -> Self;
}

impl<D: Dimension> SqrtArr for Array<f64, D> {
    fn sqrt(&self) -> Self {
        let mut ans: Array<f64, D> = Array::zeros(self.dim().clone());
        Zip::from(&mut ans)
            .and(self)
            .for_each(|a, &b| *a = b.sqrt());
        ans
    }
}

fn main() {
    let a = arr2(&[
        [ 0.0,  1.0,  4.0],
        [ 9.0, 16.0, 25.0],
    ]);

    println!("a.sqrt() = ");
    println!("{:?}", a.sqrt());
}