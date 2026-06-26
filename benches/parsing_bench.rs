use criterion::{Criterion, black_box, criterion_group, criterion_main};
use shimmer::interceptor::ToolInterceptor;
use shimmer::tool;

fn bench_interceptor_feed_token(c: &mut Criterion) {
    c.bench_function("interceptor_feed_token", |b| {
        let mut interceptor = ToolInterceptor::new(false, true);
        b.iter(|| {
            interceptor.feed_token(black_box("some text token "));
            black_box(&interceptor.buffer);
        });
    });
}

fn bench_is_idempotent_read_tools(c: &mut Criterion) {
    c.bench_function("is_idempotent_rg", |b| {
        b.iter(|| tool::is_idempotent(black_box("rg"), black_box(&[])));
    });
}

fn bench_is_idempotent_sed(c: &mut Criterion) {
    c.bench_function("is_idempotent_sed", |b| {
        b.iter(|| tool::is_idempotent(black_box("sed"), black_box(&["-n".into(), "p".into()])));
    });
}

fn bench_execute_tool_echo(c: &mut Criterion) {
    c.bench_function("execute_tool_echo", |b| {
        b.iter(|| tool::execute_tool(black_box("echo"), black_box(&["hello".into()])));
    });
}

fn bench_xml_edit_parsing(c: &mut Criterion) {
    let edit = "<edit file=\"src/main.rs\">\n<search>\nfn main() {\n</search>\n<replace>\nfn \
                main() -> Result<()> {\n</replace>\n</edit>";
    c.bench_function("xml_edit_parsing", |b| {
        b.iter(|| {
            let mut interceptor = ToolInterceptor::new(false, true);
            interceptor.feed_token(black_box(edit));
        });
    });
}

criterion_group!(
    benches,
    bench_interceptor_feed_token,
    bench_is_idempotent_read_tools,
    bench_is_idempotent_sed,
    bench_execute_tool_echo,
    bench_xml_edit_parsing,
);
criterion_main!(benches);
