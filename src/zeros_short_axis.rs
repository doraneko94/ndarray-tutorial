use ndarray::*;

trait ConvArr: Sized {
    fn conv_axis(&self, axis: Axis, size: usize) -> Option<Self>;
}

impl<D: RemoveAxis> ConvArr for Array<f64, D> {
    fn conv_axis(&self, axis: Axis, size: usize) -> Option<Self> {
        if size == 0 {
            return None;
        }

        let index = axis.index();
        if self.ndim() < index {
            return None;
        }

        let n = self.shape()[index];
        if n < size {
            return None;
        }

        let mut ans = Array::zeros(self.dim().clone())
                            .slice_axis(axis, Slice::from(size-1..))
                            .to_owned();

        let mut s = self.slice_axis(axis, Slice::from(..size-1))
                        .sum_axis(axis);

        for (mut ans_sub, (tail_sub, head_sub)) in ans.axis_iter_mut(axis)
                                                .zip(
                                                    self.axis_iter(axis).skip(size-1)
                                                        .zip(self.axis_iter(axis))
                                                )
        {
            Zip::from(&mut s)
                .and(tail_sub)
                .for_each(|a, &b| *a += b);

            Zip::from(&mut ans_sub)
                .and(&s)
                .for_each(|a, &b| *a = b);

            Zip::from(&mut s)
                .and(head_sub)
                .for_each(|a, &b| *a -= b);
        }

        Some(ans)
    }
}

fn main() {
    let a = arr2(&[
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
        [9.0, 0.0, 1.0],
    ]);

    println!(" a.conv_axis(Axis(0), 3).unwrap() = ");
    println!("{:?}", a.conv_axis(Axis(0), 3).unwrap());
    println!("************************************");
    println!(" a.conv_axis(Axis(1), 2).unwrap() = ");
    println!("{:?}", a.conv_axis(Axis(1), 2).unwrap());
}